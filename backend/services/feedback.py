import os
import math
import time
import json
import numpy as np
from fastdtw import fastdtw
from collections import defaultdict
from fastapi.responses import JSONResponse
from config import settings, logger
from constants import FilePaths, ResponseMessages, KEYPOINT_MAPPING, feature_types
from models.clova import CompletionExecutor
from services.score import read_pose, normalize_landmarks_to_range, normalize_landmarks_to_range_by_mean

completion_executor = CompletionExecutor(
    host=settings.clova_host,
    api_key=settings.service_api_key,
    request_id=settings.request_id
)
pose_cache = {}
index_map_cache = {}

def get_feedback(content: str) -> str:
    """Get Clova Studio API Response."""
    preset_text = [{"role":"system","content":"* 댄스 코치로써 주어진 입력값에 대한 적절한 피드백을 주세요."},
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


def check_if_end_consonant(word):
    # 한글 유니코드 범위: 가(0xAC00) ~ 힣(0xD7A3)
    if not word:
        raise ValueError("단어가 비어있습니다!")
    
    last_char = word[-1]
    if '가' <= last_char <= '힣':
        # 한글 문자에서 종성(받침)이 있는지 확인
        # (유니코드 코드 포인트 - 0xAC00) % 28
        # 0이면 종성이 없음 (모음으로 끝남), 그 외는 종성이 있음 (자음으로 끝남)
        code = ord(last_char) - 0xAC00
        if code % 28 == 0:
            return False
        else:
            return True
    else:
        raise ValueError(f"'{word}'은(는) 한글이 아닙니다.")

def change_angle_expression(angle):
    return (angle if abs(angle) < 180 else (360 - abs(angle)) * (-1 if angle < 0 else 1))

def calculate_two_points_angle(point1, point2):
    vector = point2 - point1
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    return angle

def calculate_three_points_angle(point1, point2, point3, eps=1e-7):
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cos_angle = dot_product / (magnitude1 * magnitude2)

    cos_angle = min(1.0-eps, max(-1.0+eps, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def calculate_two_vector_angle(v1, v2, normal, eps=1e-7):
    # 두 벡터 사이의 각도를 계산하며, v1과 v2의 상대적인 위치에 따라 부호를 결정
    dot_product = np.dot(v1, v2)

    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    cos_angle = dot_product / (magnitude1 * magnitude2)

    cos_angle = min(1.0-eps, max(-1.0+eps, cos_angle))
    angle = math.degrees(math.acos(cos_angle))

    # 부호 결정을 위한 외적
    cross_product = np.cross(v1, v2)
    if np.dot(cross_product, normal) < 0:
        angle = -angle
    
    return angle

def calculate_two_points_angle_reverse(point1, point2):
    # 어느축을 기준으로 각도를 측정할지 반대로한 two_points 각도 측정 함수
    vector = point2 - point1
    angle = math.degrees(math.atan2(vector[0], vector[1]))

    return angle

def calculate_three_points_angle_with_sign(point1, point2, point3, normal, eps=1e-7):
    # 3점 사이 각도를 측정할 때 위치관계를 기반으로 부호를 결정하는 함수
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cos_angle = dot_product / (magnitude1 * magnitude2)

    cos_angle = min(1.0-eps, max(-1.0+eps, cos_angle))
    angle = math.degrees(math.acos(cos_angle))

    # 부호 결정을 위한 외적
    cross_product = np.cross(vector1, vector2)
    if np.dot(cross_product, normal) < 0:
        angle = -angle

    return angle


def project_onto_plane(v1, v2):
    """
    v1을 법선 벡터로 하는 평면에 v2를 정사영하는 함수
    :param v1: 기준이 되는 법선 벡터 (3차원)
    :param v2: 평면에 정사영할 벡터 (3차원)
    :return: 정사영된 벡터
    """
    v1_u = v1 / np.linalg.norm(v1)  # v1을 단위 벡터로 변환
    projection = v2 - np.dot(v2, v1_u) * v1_u  # 정사영 계산
    return projection


class FramePose3D:
    all_keypoints = list(KEYPOINT_MAPPING.values())

    def __init__(self, landmarks_data):
        self.keypoints = {
            k: np.array([
                landmarks_data[i][0], landmarks_data[i][1], landmarks_data[i][2]
            ]) for i, k in enumerate(self.all_keypoints)
        }
        # neck
        self.keypoints['neck'] = (self.keypoints['mouth_left'] + self.keypoints['mouth_right']) / 2
        # hip center
        self.keypoints['pelvis'] = (self.keypoints['left_hip'] + self.keypoints['right_hip']) / 2
        # body center
        self.keypoints['torso'] = (self.keypoints['left_shoulder'] + self.keypoints['right_shoulder'] + self.keypoints['left_hip'] + self.keypoints['right_hip']) / 4
        self.keypoints['belly'] = (self.keypoints['left_shoulder']*0.2 + self.keypoints['right_shoulder']*0.2 + self.keypoints['left_hip']*0.3 + self.keypoints['right_hip']*0.3)
        self.keypoints['breast'] = (self.keypoints['belly'] + self.keypoints['neck']) / 2
        # hand point
        self.keypoints['left_hand'] = (self.keypoints['left_pinky'] + self.keypoints['left_index'] + self.keypoints['left_thumb']) / 3
        self.keypoints['right_hand'] = (self.keypoints['right_pinky'] + self.keypoints['right_index'] + self.keypoints['right_thumb']) / 3
        # waist point
        self.keypoints['left_waist'] = (self.keypoints['left_hip'] * 0.6 + self.keypoints['left_shoulder'] * 0.4)
        self.keypoints['right_waist'] = (self.keypoints['right_hip'] * 0.6 + self.keypoints['right_shoulder'] * 0.4) / 2
        self.keypoints['right_foot'] = self.keypoints['right_foot_index']
        self.keypoints['left_foot'] = self.keypoints['left_foot_index']

        # 몸이 움직임에 따라 좌표축 역할을 해줄 2개의 벡터 정의
        x_direction = (self.keypoints['left_shoulder'] - self.keypoints['right_shoulder'])
        x_direction[1] = 0.
        self.shoulder_vector = x_direction # 팔과 머리의 움직임과 연관

        x_direction = (self.keypoints['left_hip'] - self.keypoints['right_hip'])
        x_direction[1] = 0.
        self.hip_vector = x_direction # 다리의 움직임과 연관

        # A포인트는 B포인트에 가장 가까이 위치함 등의 위치정보를 얻기 위한 지정된 keypoint들
        self.POSITION_KEYPOINTS = [
            'left_hand', 'right_hand',
            'left_shoulder', 'right_shoulder',
            'left_waist', 'right_waist',
            'left_hip', 'right_hip',
            'left_elbow', 'right_elbow',
            'left_knee', 'right_knee',
            'left_foot', 'right_foot',
            'belly', 'breast'
        ]

    def swap_keypoints(self):
        for k in self.keypoints:
            if k.startswith('left'):
                name_list = k.split('_')
                part_name = name_list[-1] if len(name_list) == 2 else '_'.join(name_list[1:])
                self.keypoints[k], self.keypoints[f'right_{part_name}'] = self.keypoints[f'right_{part_name}'], self.keypoints[k]

    def get_head_angle(self):
        '''
        귀 중점, 어깨중점, 허리 중점의 3point
        - 고개가 앞/뒤로 얼마나 숙여져/젖혀져 있는지 구합니다.
        - 0 ~ 180 : 앞으로 숙임
        - 0 ~ -180 : 뒤로 젖힘
        '''
        shoulder_center = (self.keypoints['right_shoulder'] + self.keypoints['left_shoulder']) / 2
        waist_center = (self.keypoints['left_waist'] + self.keypoints['right_waist']) / 2
        ear_center = (self.keypoints['right_ear'] + self.keypoints['left_ear']) / 2
        return calculate_three_points_angle(waist_center, shoulder_center, ear_center)

    def get_eye_direction(self):
        '''
        - 시선이 카메라를 기준으로 왼쪽/오른쪽 어디를 바라보고 있는지 구합니다
        - 왼쪽으로 돌고 있으면 양수, 오른쪽으로 돌고 있으면 음수
        '''
        return calculate_two_points_angle(self.keypoints['right_ear'][[0, 2]], self.keypoints['left_ear'][[0, 2]])

    def get_waist_angle(self):
        '''
        어깨중점, 엉덩이중점, 무릎 중점의 3 point
        - 허리 굽혀진 정도 계산
        - 180에 가까울수록 펴져있고, 0에 가까울수록 접혀져있음
        '''
        shoulder_center = (self.keypoints['right_shoulder'] + self.keypoints['left_shoulder']) / 2
        hip_center = (self.keypoints['left_hip'] + self.keypoints['right_hip']) / 2
        knee_center = (self.keypoints['left_knee'] + self.keypoints['right_knee']) / 2
        return calculate_three_points_angle(shoulder_center, hip_center, knee_center)

    def get_body_direction(self):
        '''
        - 사람의 몸이 전체적으로 얼마나 뒤를 돌아 있는지 판단하기위해 엉덩이 point 2개를 이용
        - 왼쪽으로 돌고 있으면 양수, 오른쪽으로 돌고 있으면 음수
        '''
        return calculate_two_points_angle(self.keypoints['right_waist'][[0, 2]], self.keypoints['left_waist'][[0, 2]])

    def get_left_elbow_angle(self):
        '''
        - 팔꿈치가 얼마나 굽어져 있는지
        - 어께, 팔꿈치, 손목 0 ~ 180
        '''
        return calculate_three_points_angle(self.keypoints['left_shoulder'], self.keypoints['left_elbow'], self.keypoints['left_wrist'])

    def get_left_arm_height(self):
        '''
        - 왼팔이 얼마나 올라가 있는지
        - 팔이 높게 위치할수록 1에 가깝고, 낮게 위치할수록 0
        '''
        hip_to_shoulder = self.keypoints['left_hip'][1] - self.keypoints['left_shoulder'][1]
        y_max = self.keypoints['left_hip'][1]
        y_min = self.keypoints['left_shoulder'][1] - hip_to_shoulder

        value = (np.clip(self.keypoints['left_elbow'][1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)

    def get_left_hand_height(self):
        '''
        - 왼손이 얼마나 올라가 있는지
        - 팔이 높게 위치할수록 1에 가깝고, 낮게 위치할수록 0
        '''
        hip_to_shoulder = self.keypoints['left_hip'][1] - self.keypoints['left_shoulder'][1]
        y_max = self.keypoints['left_hip'][1]
        y_min = self.keypoints['left_shoulder'][1] - hip_to_shoulder

        value = (np.clip(self.keypoints['left_hand'][1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)

    def get_left_arm_dir(self):
        '''
        왼팔의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.keypoints['left_shoulder'][[0, 1]], self.keypoints['left_wrist'][[0, 1]])

    def get_right_elbow_angle(self):
        '''
        - 팔꿈치가 얼마나 굽어져 있는지
        - 어께, 팔꿈치, 손목 0 ~ 180
        '''
        return calculate_three_points_angle(self.keypoints['right_shoulder'], self.keypoints['right_elbow'], self.keypoints['right_wrist'])

    def get_right_arm_height(self):
        '''
        - 오른팔이 얼마나 올라가 있는지
        - 팔이 높게 위치할수록 1에 가깝고, 낮게 위치할수록 0
        '''
        hip_to_shoulder = self.keypoints['right_hip'][1] - self.keypoints['right_shoulder'][1]
        y_max = self.keypoints['right_hip'][1]
        y_min = self.keypoints['right_shoulder'][1] - hip_to_shoulder

        value = (np.clip(self.keypoints['right_elbow'][1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)
    
    def get_right_hand_height(self):
        '''
        - 오른손이 얼마나 올라가 있는지
        - 팔이 높게 위치할수록 1에 가깝고, 낮게 위치할수록 0
        '''
        hip_to_shoulder = self.keypoints['right_hip'][1] - self.keypoints['right_shoulder'][1]
        y_max = self.keypoints['right_hip'][1]
        y_min = self.keypoints['right_shoulder'][1] - hip_to_shoulder

        value = (np.clip(self.keypoints['right_hand'][1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)


    def get_right_arm_dir(self):
        '''
        오른팔의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.keypoints['right_shoulder'][[0, 2]], self.keypoints['right_wrist'][[0, 2]])

    def get_left_knee_angle(self):
        '''
        - 무릎이 얼마나 굽어져 있는지지
        - 엉덩이, 무릎, 발목 0 ~ 180
        '''
        return calculate_three_points_angle(self.keypoints['left_hip'], self.keypoints['left_knee'], self.keypoints['left_ankle'])

    def get_left_leg_height(self):
        '''
        - 골반을 기준으로 왼다리를 얼마나 올라가 있는지.
        - 아래에 위치할수록 0, 위로갈수록 1
        '''
        hip_to_ground = 0.9999 - self.keypoints['left_hip'][1]
        y_max = 0.9999
        y_min = self.keypoints['left_hip'][1] - hip_to_ground

        value = (np.clip(self.keypoints['left_ankle'][1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)

    def get_left_leg_dir(self):
        '''
        왼다리의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.keypoints['left_hip'][[0, 1]], self.keypoints['left_knee'][[0, 1]])

    def get_right_knee_angle(self):
        '''
        - 무릎이 얼마나 굽어져 있는지지
        - 엉덩이, 무릎, 발목 0 ~ 180
        '''
        return calculate_three_points_angle(self.keypoints['right_hip'], self.keypoints['right_knee'], self.keypoints['right_ankle'])

    def get_right_leg_height(self):
        '''
        - 골반을 기준으로 오른다리가 얼마나 올라가 있는지.
        - 아래에 위치할수록 0, 위로갈수록 1
        '''
        hip_to_ground = 0.9999 - self.keypoints['right_hip'][1]
        y_max = 0.9999
        y_min = self.keypoints['right_hip'][1] - hip_to_ground

        value = (np.clip(self.keypoints['right_ankle'][1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)

    def get_right_leg_dir(self):
        '''
        오른다리의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.keypoints['right_hip'][[0, 1]], self.keypoints['right_knee'][[0, 1]])

    def compute_distance(self, point1, point2, idx=[0,1,2]):
        """ 두 점 사이의 L2 거리(유클리드 거리) 계산 """
        return np.linalg.norm(self.keypoints[point1][idx] - self.keypoints[point2][idx])

    def get_closest_keypoint(self, point):
        if point == 'left_hand':
            skip = ['left_elbow', 'left_hand', 'right_hand']
        elif point == 'right_hand':
            skip = ['right_elbow', 'left_hand', 'right_hand']
        elif point == 'left_foot':
            skip = ['left_knee', 'left_foot', 'right_foot']
        elif point == 'right_foot':
            skip = ['right_knee', 'left_foot', 'right_foot']
        else:
            skip = []
        
        min_distance = self.compute_distance(point, 'left_waist', [0, 1])
        min_keypoint = 'left_waist'
        for k in self.POSITION_KEYPOINTS:
            if k in skip:
                continue
            new_distance = self.compute_distance(point, k, [0, 1])
            if new_distance < min_distance:
                min_distance = new_distance
                min_keypoint = k
        
        return min_keypoint, min_distance

def refine_float_dict(differences):
    for part in differences:
        for k, v in differences[part].items():
            if isinstance(v, dict):
                for feature in v:
                    if isinstance(v[feature], str):
                        continue

                    differences[part][k][feature] = f'{differences[part][k][feature]:.4f}'
                    differences[part][k][feature] = float(differences[part][k][feature])
            else:
                if feature_types[part][k] == int:
                    differences[part][k] = int(differences[part][k])
                else:
                    differences[part][k] = f'{differences[part][k]:.4f}'
                    differences[part][k] = float(differences[part][k])

    return differences


def get_difference_dict(pose1, pose2, reverse=True):
    if reverse:
        pose1.swap_keypoints()
        pose2.swap_keypoints()

    ### 왼손의 포지션 결정
    target_min_keypoint, target_dist = pose1.get_closest_keypoint('left_hand')
    user_min_keypoint, user_dist = pose2.get_closest_keypoint('left_hand')
    user_dist = pose2.compute_distance('left_hand', target_min_keypoint)
    left_hand_info = {
        'target_keypoint': target_min_keypoint,
        'user_keypoint': user_min_keypoint,
        'pose1': target_dist,
        'pose2': user_dist,
        'diff': target_dist-user_dist
    }

    ### 오른손의 포지션 결정
    target_min_keypoint, target_dist = pose1.get_closest_keypoint('right_hand')
    user_min_keypoint, user_dist = pose2.get_closest_keypoint('right_hand')
    user_dist = pose2.compute_distance('right_hand', target_min_keypoint)
    right_hand_info = {
        'target_keypoint': target_min_keypoint,
        'user_keypoint': user_min_keypoint,
        'pose1': target_dist,
        'pose2': user_dist,
        'diff': target_dist-user_dist
    }

    result = {
        'head':{
            'lower_angle_difference': change_angle_expression(int(pose1.get_head_angle() - pose2.get_head_angle())), # 음수인 경우 고개를 좀 더 숙여라, 양수인 경우 고개를 좀 더 들어라
            'direction_difference': change_angle_expression(int(pose1.get_eye_direction() - pose2.get_eye_direction())) # 음수인 경우 좀 더 오른쪽을 바라봐라, 양수인 경우 좀 더 왼쪽을 바라봐라
        },
        'body':{
            'bend_angle_difference': int(pose1.get_waist_angle() - pose2.get_waist_angle()), # 음수면 허리를 더 굽혀라, 양수면 허리를 더 펴라
            'direction_difference': change_angle_expression(int(pose1.get_body_direction() - pose2.get_body_direction())) # 음수면 몸을 더 오른쪽으로 돌려라, 양수면 몸을 더 왼쪽으로 돌려라
        },
        'left_arm':{
            'bend_angle_difference': int(pose1.get_left_elbow_angle() - pose2.get_left_elbow_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'arm_height_difference': int(180 * (pose1.get_left_arm_height() - pose2.get_left_arm_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'hand_height_difference': int(180 * (pose1.get_left_hand_height() - pose2.get_left_hand_height())), # 음수면 왼손을 더 내려라, 양수면 왼손을 더 올려라
            'direction_difference': change_angle_expression(int(pose1.get_left_arm_dir() - pose2.get_left_arm_dir())), # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
            'closest_point_difference': left_hand_info
        },
        'right_arm':{
            'bend_angle_difference': int(pose1.get_right_elbow_angle() - pose2.get_right_elbow_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'arm_height_difference': int(180 * (pose1.get_right_arm_height() - pose2.get_right_arm_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'hand_height_difference': int(180 * (pose1.get_right_hand_height() - pose2.get_right_hand_height())), # 음수면 오른팔을 더 내려라, 양수면 오른팔을 더 올려라
            'direction_difference': change_angle_expression(int(pose1.get_right_arm_dir() - pose2.get_right_arm_dir())), # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
            'closest_point_difference': right_hand_info
        },
        'left_leg':{
            'bend_angle_difference': int(pose1.get_left_knee_angle() - pose2.get_left_knee_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'height_difference': int(180 * (pose1.get_left_leg_height() - pose2.get_left_leg_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'direction_difference': change_angle_expression(int(pose1.get_left_leg_dir() - pose2.get_left_leg_dir())) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
        },
        'right_leg':{
            'bend_angle_difference': int(pose1.get_right_knee_angle() - pose2.get_right_knee_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'height_difference': int(180 * (pose1.get_right_leg_height() - pose2.get_right_leg_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'direction_difference': change_angle_expression(int(pose1.get_right_leg_dir() - pose2.get_right_leg_dir())) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
        },
        'leg':{
            'knee_distance_difference': (pose1.compute_distance('left_knee', 'right_knee') - pose2.compute_distance('left_knee', 'right_knee')), # 음수면 무릎을 더 붙여라, 양수면 무릎을 너무 붙이지 마라
            'foot_distance_difference': (pose1.compute_distance('left_foot', 'right_foot') - pose2.compute_distance('left_foot', 'right_foot'))
        }
    }
    result = refine_float_dict(result)
    return json.dumps(result)


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

        # 정규화
        normalized_all_frame_points2 = normalize_landmarks_to_range_by_mean(all_frame_points1, all_frame_points2)
        
        # Clova API를 이용한 원본 영상 프레임과 유저 영상 프레임 포즈 피드백 받기
        start_time = time.time()
        points1 = all_frame_points1[target_frame]
        points2 = normalized_all_frame_points2[user_frame]
        feedback = ResponseMessages.FEEDBACK_POSE_FAIL.value
        
        if np.any(points1 != (-1, -1, -1)) and np.any(points2 != (-1, -1, -1)):
            pose1 = FramePose3D(points1)
            pose2 = FramePose3D(points2)

            content = get_difference_dict(pose1, pose2, reverse=True)
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
