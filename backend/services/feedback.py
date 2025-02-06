import os
import math
import json
import h5py
import numpy as np
from fastdtw import fastdtw
from collections import defaultdict
from fastapi.responses import JSONResponse
from config import settings
from constants import FilePaths
from models.clova import CompletionExecutor
from services.score import normalize_landmarks_to_range

completion_executor = CompletionExecutor(
    host=settings.clova_host,
    api_key=settings.service_api_key,
    request_id=settings.request_id
)
pose_cache = {}
index_map_cache = {}

def read_pose(h5_path):
    try:
        with h5py.File(h5_path, "r") as f:
            width = f["width"][()]
            height = f["height"][()]
            all_frame_points = f["all_frames_points"][()]

        return width, height, all_frame_points
    except Exception as e:
        raise ValueError(f"Failed to read pose data from {h5_path}: {str(e)}")

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

    def get_angle(self, key1, key2):
        p1 = getattr(self, key1) 
        p2 = getattr(self, key2)

        vector = p2 - p1
        angle = math.degrees(math.atan2(vector[1], vector[0]))

        return angle

    def get_joint_angle(self, key1, key2, key3):
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

def get_content(pose1, pose2) -> str:
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

def get_feedback(content):
    preset_text = [{"role":"system","content":"{\r\n    \"ANGLE DIFFERENCES\": {\r\n        \"Explation\": \"**모범 포즈와 사용자의 포즈에서 ANGLE_FEATURES를 각각 구해 그 차이를 계산한 값.**\",\r\n        \"head difference\": [\"{'target face angle'-'user face angle'}\", \"양수값일 경우 'target face angle'보다 user의 머리가 더 오른쪽으로 기울어져 있다는 뜻이다.\"],\r\n        \"shoulder difference\": [\"{'target shoulder angle'-'user shoulder angle'}\", \"양수값일 경우 'target shoulder angle'보다 user의 어깨가 더 오른쪽으로 기울어져 있다는 뜻이다.\"],\r\n        \"left arm angle difference\": [\"{'target left arm angle'-'user left arm angle'}\", \"양수값일 경우 'target left arm angle'보다 user의 왼팔이 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"right arm angle difference\": [\"{'target right arm angle'-'user right arm angle'}\", \"양수값일 경우 'target right arm angle'보다 user의 오른팔이 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"left elbow angle difference\": [{'target left elbow angle'-'user left elbow angle'}, \"양수값일 경우 'target left elbow angle'보다 user의 왼팔이 더 굽혀져있다는 뜻이다.\"],\r\n        \"right elbow angle difference\": [{'target right elbow angle'-'user right elbow angle'}, \"양수값일 경우 'target right elbow angle'보다 user의 오른팔이 더 굽혀져있다는 뜻이다.\"],\r\n        \"left leg angle difference\": [\"{'target left leg angle'-'target right leg angle'}\", \"양수값일 경우 'target left leg angle'보다 user의 왼다리가 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"right leg angle difference\": [\"{'user right leg angle'-'target right leg angle'}\", \"양수값일 경우 'target right leg angle'보다 user의 오른다리가 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"left knee angle difference\": [\"{'target left knee angle'-'user left knee angle'}\", \"양수값의 경우 'target left knee angle'보다 user의 왼다리가 더 굽혀져있다는 뜻이다.\"],\r\n        \"right knee angle difference\": [\"{'target right knee angle'-'user right knee angle'}\", \"양수값의 경우 'target right knee angle'보다 user의 오른다리가 더 굽혀져있다는 뜻이다.\"]\r\n    },\r\n    \"ROLE\": '''\r\n        당신은 서로 다른 두 사람의 Pose Difference 정보를 기반으로 피드백을 주는 댄스 서포터 AI입니다.\r\n        입력값으로는 두 사람 포즈의 차이에 대한 설명이 총 10가지 주어집니다.\r\n        * difference의 절댓값이 30 이상일 경우에 대해서만 피드백을 주도록 합니다. difference의 절댓값이 30 이하의 경우에는 피드백을 전달하지 않도록 합니다.\r\n        * 구체적인 수치를 나타내기보다는, 단순한 문장으로 바꾸어 표현합니다.(예시 - 왼쪽 팔꿈치는 70도만큼 덜 굽혀져 있습니다 -> 왼쪽 팔꿈치를 더 펴주세요.)\r\n    '''\r\n *모든 차이값의 절대값이 30이하여서 피드백을 줄 필요가 없을 때에는 '완벽합니다!'를 출력합니다.}"},
                    {"role":"user","content":content}]
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
    folder_id = request.folder_id
    if folder_id not in pose_cache:
        root_path = os.path.join("data", folder_id)
        target_path = os.path.join(root_path, FilePaths.ORIGIN_H5.value)
        user_path = os.path.join(root_path, FilePaths.USER_H5.value)

        _, _, all_frame_points1 = read_pose(target_path)
        _, _, all_frame_points2 = read_pose(user_path)
        pose_cache[folder_id] = (all_frame_points1, all_frame_points2)

    all_frame_points1, all_frame_points2 = pose_cache.get(folder_id, ([], []))

    if folder_id not in index_map_cache:
        index_map = defaultdict(list)
        _, pairs = fastdtw(all_frame_points1, all_frame_points2, dist=normalize_landmarks_to_range)
        for i in pairs:
            index_map[i[1]].append(i[0])
        index_map_cache[folder_id] = index_map

    index_map = index_map_cache.get(folder_id, {})
    user_frame = int(request.frame)
    target_frame = index_map.get(user_frame, [0])[0]

    points1 = all_frame_points1[target_frame]
    points2 = all_frame_points2[user_frame]
    if np.any(points1 == (-1, -1, -1)) or np.any(points2 == (-1, -1, -1)):
        return JSONResponse(content={"feedback": "포즈가 감지되지 않아 피드백을 드릴 수 없습니다 (┬┬﹏┬┬)"}, status_code=200)
    
    pose1 = FramePose(points1)
    pose2 = FramePose(points2)

    content = get_content(pose1, pose2)
    feedback = get_feedback(content=content)

    return JSONResponse(content={"feedback": feedback}, status_code=200)

async def clear_cache_and_files(folder_id: str):
    if folder_id in pose_cache:
        del pose_cache[folder_id]
    if folder_id in index_map_cache:
        del index_map_cache[folder_id]

    folder_path = os.path.join("data", folder_id)
    origin_path = os.path.join(folder_path, FilePaths.ORIGIN_MP4.value)
    user_path = os.path.join(folder_path, FilePaths.USER_MP4.value)
    os.remove(origin_path)
    os.remove(user_path)
    
    return {"message": f"Cache cleared for folder_id: {folder_id}"}