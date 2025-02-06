import os
import math
import time
import json
import numpy as np
from fastdtw import fastdtw
from collections import defaultdict
from fastapi.responses import JSONResponse
from config import settings, logger
from constants import FilePaths, ResponseMessages
from models.clova import CompletionExecutor
from services.score import read_pose, normalize_landmarks_to_range

completion_executor = CompletionExecutor(
    host=settings.clova_host,
    api_key=settings.service_api_key,
    request_id=settings.request_id
)
pose_cache = {}
index_map_cache = {}

class FramePose:
    keypoints = [
        'nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    def __init__(self, points):
        for i, key in enumerate(self.keypoints):
            setattr(self, key, np.array([points[i][0], points[i][1]]))

    def get_angle(self, key1: str, key2: str) -> float:
        p1 = getattr(self, key1) 
        p2 = getattr(self, key2)

        vector = p2 - p1
        angle = math.degrees(math.atan2(vector[1], vector[0]))

        return angle

    def get_joint_angle(self, key1: str, key2: str, key3: str) -> float:
        p1 = getattr(self, key1)
        p2 = getattr(self, key2)
        p3 = getattr(self, key3)

        vector1 = p1 - p2
        vector2 = p3 - p2
        dot_product = np.dot(vector1, vector2)

        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle = math.degrees(math.acos(cos_angle))

        return angle

def get_content(pose1: FramePose, pose2: FramePose) -> str:
    """Get Clova Studio API Prompt Content."""
    angle_differences = {
        'head_difference': pose1.get_angle('right_ear', 'left_ear') - pose2.get_angle('right_ear', 'left_ear'),
        'shoulder_difference': pose1.get_angle('right_shoulder', 'left_shoulder') - pose2.get_angle('right_shoulder', 'left_shoulder'),
        'left_arm_angle_difference': pose1.get_angle('left_shoulder', 'left_elbow') - pose2.get_angle('left_shoulder', 'left_elbow'),
        'right_arm_angle_difference': pose1.get_angle('right_shoulder', 'right_elbow') - pose2.get_angle('right_shoulder', 'right_elbow'),
        'left_elbow_angle_difference': pose1.get_joint_angle('left_shoulder', 'left_elbow', 'left_wrist') - pose2.get_joint_angle('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow_angle_difference': pose1.get_joint_angle('right_shoulder', 'right_elbow', 'right_wrist') - pose2.get_joint_angle('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_leg_angle_difference': pose1.get_angle('left_hip', 'left_knee') - pose2.get_angle('left_hip', 'left_knee'),
        'right_leg_angle_difference': pose1.get_angle('right_hip', 'right_knee') - pose2.get_angle('right_hip', 'right_knee'),
        'left_knee_angle_difference': pose1.get_joint_angle('left_hip', 'left_knee', 'left_ankle') - pose2.get_joint_angle('left_hip', 'left_knee', 'left_ankle'),
        'right_knee_angle_difference': pose1.get_joint_angle('right_hip', 'right_knee', 'right_ankle') - pose2.get_joint_angle('right_hip', 'right_knee', 'right_ankle'),
    }

    return json.dumps({key: int(value) for key, value in angle_differences.items()})

def get_feedback(content: str) -> str:
    """Get Clova Studio API Response."""
    preset_text = [{"role":"system","content":"{\r\n    \"IMPORTANT_RULES\": {\r\n        \"1\": \"욕설, 비속어, 혐오표현을 사용한 문장은 생성하지 않도록 합니다.\",\r\n        \"2\": \"친절한 말투로 사용자를 지도하도록 합니다.\",\r\n        \"3\": \"사용자에게 지도할 때 구어체를 사용하여 친밀함을 느낄 수 있도록 합니다.\"\r\n    },\r\n    \"ROLE\": [\r\n        \"당신은 서로 다른 두 사람의 Pose Difference 정보를 기반으로 피드백을 주는 댄스 서포터 AI입니다.\",\r\n        \"입력값으로는 두 사람 포즈의 차이에 대한 설명이 총 10가지 주어지며, 당신은 이를 분석해서 사용자가 목표자세를 정확히 따라하기 위한 적절한 피드백을 주어야합니다.\",\r\n        \"* difference의 절댓값이 30 이상일 경우에 대해서만 피드백을 주도록 합니다. difference의 절댓값이 30 이하의 경우에는 피드백을 전달하지 않도록 합니다.\",\r\n        \"* 구체적인 수치를 나타내기보다는, 단순한 문장으로 바꾸어 표현합니다.(예시 - 왼쪽 팔꿈치는 70도만큼 덜 굽혀져 있습니다 -> 왼쪽 팔꿈치를 더 펴주세요.)\"\r\n    ],\r\n    \"INFORMATION\": {\r\n        \"head_difference\": \"사용자 머리가 목표 자세보다 더 오른쪽으로 기울어져 있는 경우 양수, 더 왼쪽으로 기울어져 있는 경우 음수이다.\",\r\n        \"shoulder_difference\": \"사용자 어깨가 목표 자세보다 더 오른쪽으로 기울어져 있는 경우 양수, 더 왼쪽으로 기울어져 있는 경우 음수이다.\",\r\n        \"arm_angle_difference\": \"사용자 팔이 목표 자세보다 상대적으로 왼쪽에 위치하는 경우 양수, 상대적으로 오른쪽에 위치하는 경우 음수이다.\",\r\n        \"elbow_angle_difference\": \"사용자 팔꿈치가 목표자세보다 더 굽혀져 있을 경우 양수, 더 펴져있을 경우 음수이다.\",\r\n        \"leg_angle_difference\": \"사용자 다리가 목표 자세보다 상대적으로 왼쪽에 위치하는 경우 양수, 상대적으로 오른쪽에 위치하는 경우 음수이다.\",\r\n        \"knee_angle_difference\": \"사용자 무릎이 목표자세보다 더 굽혀져 있을 경우 양수, 더 펴져있을 경우 음수이다.\"\r\n    },\r\n    \"EXAMPLE\": {\r\n        \"1\":{\r\n            \"inputs\": \"{'head_difference': -12, 'shoulder_difference': -19, 'left_arm_angle_difference': 10, 'right_arm_angle_difference': -3, 'left_elbow_angle_difference': 26, 'right_elbow_angle_difference': 22, 'left_leg_angle_difference': -11, 'right_leg_angle_difference': 14, 'left_knee_angle_difference': 12, 'right_knee_angle_difference': 16}\",\r\n            \"thoughts\": \"모든 동작에 대해서 목표 자세와 사용자 자세의 차이 절대값이 30보다 작으므로 칭찬을 해 주어야겠다.\"\r\n            \"outputs\": \"와! 흐름이 대단한데요! 이 느낌 그대로 이어서 다음 동작도 멋지게 해 봐요!\"\r\n        },\r\n        \"2\":{\r\n            \"inputs\": \"{'head_difference': 10, 'shoulder_difference': -17, 'left_arm_angle_difference': -36, 'right_arm_angle_difference': -25, 'left_elbow_angle_difference': 43, 'right_elbow_angle_difference': 49, 'left_leg_angle_difference': -5, 'right_leg_angle_difference': 1, 'left_knee_angle_difference': -18, 'right_knee_angle_difference': -1}\",\r\n            \"thoughts\": \"왼쪽 팔, 왼쪽 팔꿈치, 오른쪽 팔꿈치에 대해서 임계치 이상의 difference가 확인되니까 이 부분에 대해 피드백을 주어야겠다.\"\r\n            \"outputs\": \"네, 동작 차이를 기반으로 피드백 드릴게요! 왼쪽 팔을 왼쪽으로 조금 더 돌려 주시고, 왼쪽 팔꿈치와 오른쪽 팔꿈치 모두 조금 더 펴 주세요. 벌써 완벽에 가까워지고 있네요! 우리 같이 계속 노력해 봐요!\"\r\n        }\r\n    }\r\n}"},{"role":"user","content":"{'head_difference': -27, 'shoulder_difference': -16, 'left_arm_angle_difference': 25, 'right_arm_angle_difference': -14, 'left_elbow_angle_difference': 8, 'right_elbow_angle_difference': -27, 'left_leg_angle_difference': -44, 'right_leg_angle_difference': 17, 'left_knee_angle_difference': -44, 'right_knee_angle_difference': 28}"},
                    {"role":"user", "content":content}]
    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1024,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    result_dict = completion_executor.execute(request_data)
    result_dict = json.loads(result_dict)
    feedback = result_dict['message']['content']

    return feedback

async def get_frame_feedback_service(request):
    try:
        folder_id = request.folder_id
        if folder_id not in pose_cache:
            root_path = os.path.join("data", folder_id)
            target_path = os.path.join(root_path, FilePaths.ORIGIN_H5.value)
            user_path = os.path.join(root_path, FilePaths.USER_H5.value)

            # 원본 영상 및 유저 영상 포즈 정보 읽기
            _, _, all_frame_points1 = read_pose(target_path)
            _, _, all_frame_points2 = read_pose(user_path)
            pose_cache[folder_id] = (all_frame_points1, all_frame_points2)

        all_frame_points1, all_frame_points2 = pose_cache.get(folder_id, ([], []))

        if folder_id not in index_map_cache:
            # DTW를 이용한 유저 영상 프레임에 대응하는 원본 영상 프레임 추출
            index_map = defaultdict(list)
            _, pairs = fastdtw(all_frame_points1, all_frame_points2, dist=normalize_landmarks_to_range)
            for i in pairs:
                index_map[i[1]].append(i[0])
            index_map_cache[folder_id] = index_map

        index_map = index_map_cache.get(folder_id, {})
        user_frame = int(request.frame)
        target_frame = index_map.get(user_frame, [0])[0]

        # Clova API를 이용한 원본 영상 프레임과 유저 영상 프레임 포즈 피드백 받기
        start_time = time.time()
        points1 = all_frame_points1[target_frame]
        points2 = all_frame_points2[user_frame]

        feedback = ResponseMessages.FEEDBACK_POSE_FAIL.value
        
        if np.any(points1 != (-1, -1, -1)) and np.any(points2 != (-1, -1, -1)):
            pose1 = FramePose(points1)
            pose2 = FramePose(points2)

            content = get_content(pose1, pose2)
            feedback = get_feedback(content=content)
        end_time = time.time()

        logger.info(f"[{folder_id}] get pose feedback success: {end_time - start_time} sec")
        return JSONResponse(content={"feedback": feedback, "frame": target_frame}, status_code=200)

    except Exception as e:
        logger.error(f"[{folder_id}] get pose feedback fail: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

async def clear_cache_and_files(folder_id: str):
    try:
        # 포즈 정보 및 DTW 정보 캐시 정리
        if folder_id in pose_cache:
            del pose_cache[folder_id]
        if folder_id in index_map_cache:
            del index_map_cache[folder_id]

        # 원본 영상 및 유저 영상 삭제
        folder_path = os.path.join("data", folder_id)
        origin_path = os.path.join(folder_path, FilePaths.ORIGIN_MP4.value)
        user_path = os.path.join(folder_path, FilePaths.USER_MP4.value)
        os.remove(origin_path)
        os.remove(user_path)

        logger.info(f"[{folder_id}] clear cache and files success")
        return JSONResponse(content={"message": ResponseMessages.FEEDBACK_CLEAR_SUCCESS.value.format(folder_id)}, status_code=200)

    except Exception as e:
        logger.error(f"[{folder_id}] clear cache and files fail: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
