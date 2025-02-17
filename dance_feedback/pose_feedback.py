import math
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

body_parts_korean = {
    "nose": "코",
    "left_eye_inner": "왼쪽 눈 안쪽",
    "left_eye": "왼쪽 눈",
    "left_eye_outer": "왼쪽 눈 바깥쪽",
    "right_eye_inner": "오른쪽 눈 안쪽",
    "right_eye": "오른쪽 눈",
    "right_eye_outer": "오른쪽 눈 바깥쪽",
    "left_ear": "왼쪽 귀",
    "right_ear": "오른쪽 귀",
    "mouth_left": "왼쪽 입",
    "mouth_right": "오른쪽 입",
    "left_shoulder": "왼쪽 어깨",
    "right_shoulder": "오른쪽 어깨",
    "left_elbow": "왼쪽 팔꿈치",
    "right_elbow": "오른쪽 팔꿈치",
    "left_wrist": "왼쪽 손목",
    "right_wrist": "오른쪽 손목",
    "left_pinky": "왼쪽 새끼손가락",
    "right_pinky": "오른쪽 새끼손가락",
    "left_index": "왼쪽 검지",
    "right_index": "오른쪽 검지",
    "left_thumb": "왼쪽 엄지",
    "right_thumb": "오른쪽 엄지",
    "left_hip": "왼쪽 골반",
    "right_hip": "오른쪽 골반",
    "left_knee": "왼쪽 무릎",
    "right_knee": "오른쪽 무릎",
    "left_ankle": "왼쪽 발목",
    "right_ankle": "오른쪽 발목",
    "left_heel": "왼쪽 발뒤꿈치",
    "right_heel": "오른쪽 발뒤꿈치",
    "left_foot_index": "왼쪽 발가락",
    "right_foot_index": "오른쪽 발가락",
    "eye": "눈",
    "ear": "귀",
    "mouth": "입",
    "shoulder": "어깨",
    "elbow": "팔꿈치",
    "wrist": "손목",
    "pinky": "새끼손가락",
    "index": "검지",
    "thumb": "엄지",
    "hip": "엉덩이",
    "knee": "무릎",
    "ankle": "발목",
    "heel": "발뒤꿈치",
    "foot_index": "발가락",
    "left_hand": "왼손",
    "right_hand": "오른손",
    "hand": "손",
    "right_foot": "오른발",
    "left_foot": "왼발",
    "foot": "발",
    "torso": "몸",
    'waist': "허리",
    "left_waist": "왼쪽 허리",
    "right_waist": "오른쪽 허리",
    "arm": "팔",
    "left_arm": "왼팔",
    "right_arm": "오른팔",
    "leg": "다리",
    "left_leg": "왼쪽 다리",
    "right_leg": "오른쪽 다리",
    "belly": "배",
    "head": "머리",
    "breast": "가슴"
}

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
    # 두 벡터 사이의 각도를 계산하며, v1과 v2의 상대적인 위치에 따라 부호를 결정합니다.
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
    def __init__(self, landmarks_data):
        self.keypoints = {
            "nose": np.array([
                landmarks_data['head']['0']['x'],
                landmarks_data['head']['0']['y'],
                landmarks_data['head']['0']['z']
            ]),
            "left_eye": np.array([
                landmarks_data['head']['2']['x'],
                landmarks_data['head']['2']['y'],
                landmarks_data['head']['2']['z']
            ]),
            "right_eye": np.array([
                landmarks_data['head']['5']['x'],
                landmarks_data['head']['5']['y'],
                landmarks_data['head']['5']['z']
            ]),
            "left_ear": np.array([
                landmarks_data['head']['7']['x'],
                landmarks_data['head']['7']['y'],
                landmarks_data['head']['7']['z']
            ]),
            "right_ear": np.array([
                landmarks_data['head']['8']['x'],
                landmarks_data['head']['8']['y'],
                landmarks_data['head']['8']['z']
            ]),
            "left_mouth": np.array([
                landmarks_data['head']['9']['x'],
                landmarks_data['head']['9']['y'],
                landmarks_data['head']['9']['z']
            ]),
            "right_mouth": np.array([
                landmarks_data['head']['10']['x'],
                landmarks_data['head']['10']['y'],
                landmarks_data['head']['10']['z']
            ]),
            "left_shoulder": np.array([
                landmarks_data['left_arm']['11']['x'],
                landmarks_data['left_arm']['11']['y'],
                landmarks_data['left_arm']['11']['z']
            ]),
            "right_shoulder": np.array([
                landmarks_data['right_arm']['12']['x'],
                landmarks_data['right_arm']['12']['y'],
                landmarks_data['right_arm']['12']['z']
            ]),
            "left_elbow": np.array([
                landmarks_data['left_arm']['13']['x'],
                landmarks_data['left_arm']['13']['y'],
                landmarks_data['left_arm']['13']['z']
            ]),
            "right_elbow": np.array([
                landmarks_data['right_arm']['14']['x'],
                landmarks_data['right_arm']['14']['y'],
                landmarks_data['right_arm']['14']['z']
            ]),
            "left_wrist": np.array([
                landmarks_data['left_arm']['15']['x'],
                landmarks_data['left_arm']['15']['y'],
                landmarks_data['left_arm']['15']['z']
            ]),
            "right_wrist": np.array([
                landmarks_data['right_arm']['16']['x'],
                landmarks_data['right_arm']['16']['y'],
                landmarks_data['right_arm']['16']['z']
            ]),
            "left_pinky": np.array([
                landmarks_data['left_arm']['17']['x'],
                landmarks_data['left_arm']['17']['y'],
                landmarks_data['left_arm']['17']['z']
            ]),
            "right_pinky": np.array([
                landmarks_data['right_arm']['18']['x'],
                landmarks_data['right_arm']['18']['y'],
                landmarks_data['right_arm']['18']['z']
            ]),
            "left_index": np.array([
                landmarks_data['left_arm']['19']['x'],
                landmarks_data['left_arm']['19']['y'],
                landmarks_data['left_arm']['19']['z']
            ]),
            "right_index": np.array([
                landmarks_data['right_arm']['20']['x'],
                landmarks_data['right_arm']['20']['y'],
                landmarks_data['right_arm']['20']['z']
            ]),
            "left_thumb": np.array([
                landmarks_data['left_arm']['21']['x'],
                landmarks_data['left_arm']['21']['y'],
                landmarks_data['left_arm']['21']['z']
            ]),
            "right_thumb": np.array([
                landmarks_data['right_arm']['22']['x'],
                landmarks_data['right_arm']['22']['y'],
                landmarks_data['right_arm']['22']['z']
            ]),
            "left_hip": np.array([
                landmarks_data['left_leg']['23']['x'],
                landmarks_data['left_leg']['23']['y'],
                landmarks_data['left_leg']['23']['z']
            ]),
            "right_hip": np.array([
                landmarks_data['right_leg']['24']['x'],
                landmarks_data['right_leg']['24']['y'],
                landmarks_data['right_leg']['24']['z']
            ]),
            "left_knee": np.array([
                landmarks_data['left_leg']['25']['x'],
                landmarks_data['left_leg']['25']['y'],
                landmarks_data['left_leg']['25']['z']
            ]),
            "right_knee": np.array([
                landmarks_data['right_leg']['26']['x'],
                landmarks_data['right_leg']['26']['y'],
                landmarks_data['right_leg']['26']['z']
            ]),
            "left_ankle": np.array([
                landmarks_data['left_leg']['27']['x'],
                landmarks_data['left_leg']['27']['y'],
                landmarks_data['left_leg']['27']['z']
            ]),
            "right_ankle": np.array([
                landmarks_data['right_leg']['28']['x'],
                landmarks_data['right_leg']['28']['y'],
                landmarks_data['right_leg']['28']['z']
            ]),
            "left_foot": np.array([
                landmarks_data['left_foot']['31']['x'],
                landmarks_data['left_foot']['31']['y'],
                landmarks_data['left_foot']['31']['z']
            ]),
            "right_foot": np.array([
                landmarks_data['right_foot']['32']['x'],
                landmarks_data['right_foot']['32']['y'],
                landmarks_data['right_foot']['32']['z']
            ])
        }
        # 기존 키포인트로부터 새로운 좌표 정의
        # neck
        self.keypoints['neck'] = (self.keypoints['left_mouth'] + self.keypoints['right_mouth']) / 2
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
                part_name = k.split('_')[-1]
                self.keypoints[k], self.keypoints[f'right_{part_name}'] = self.keypoints[f'right_{part_name}'], self.keypoints[k]


    #################################### HEAD
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

    #################################### BODY
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

    #################################### LEFT ARM
    def get_left_elbow_angle(self):
        # 왼팔 굽혀진 정도. 0 ~ 180
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

    #################################### RIGHT ARM
    def get_right_elbow_angle(self):
        # 오른팔 굽혀진 정도. 0 ~ 180
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

    #################################### LEFT LEG
    def get_left_knee_angle(self):
        # 왼다리 굽혀진 정도. 0 ~ 180
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

    #################################### RIGHT LEG
    def get_right_knee_angle(self):
        # 오른다리 굽혀진 정도. 0 ~ 180
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



    ################################################## EXTRA
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

def get_difference_dict(target_landmarks_json_path, compare_landmarks_json_path, reverse=False):
    if isinstance(target_landmarks_json_path, str):
        with open(target_landmarks_json_path, 'r') as f:
            data1 = json.load(f)
    else:
        data1 = target_landmarks_json_path
    
    if isinstance(compare_landmarks_json_path, str):
        with open(compare_landmarks_json_path, 'r') as f:
            data2 = json.load(f)
    else:
        data2 = compare_landmarks_json_path

    pose1 = FramePose3D(data1)
    pose2 = FramePose3D(data2)

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
    return result

# 피드백 문구 템플릿
templates = {
    'head': {
        'lower_angle_difference': ('머리를 좀 더 숙이세요.', '머리를 너무 숙이지 마세요.'),
        'direction_difference': ('머리를 더 오른쪽으로 돌리세요.', '머리를 더 왼쪽으로 돌리세요.')
    },
    'body': {
        'bend_angle_difference': ('허리를 더 굽히세요.', '허리를 더 펴세요.'),
        'direction_difference': ('몸을 더 오른쪽으로 돌리세요.', '몸을 더 왼쪽으로 돌리세요.')
    },
    'left_arm': {
        'bend_angle_difference': ('왼팔을 더 굽히세요.', '왼팔을 더 펴주세요.'),
        'arm_height_difference': ('왼팔을 더 내려주세요.', '왼팔을 더 올려주세요.'),
        'hand_height_difference': ('왼손을 더 내려야 합니다.', '왼손을 더 올려야 합니다.'),
        'direction_difference': ('왼팔 방향을 맞춰주세요.', '왼팔 방향을 맞춰주세요.'),
    },
    'right_arm': {
        'bend_angle_difference': ('오른팔을 더 굽히세요.', '오른팔을 더 펴주세요.'),
        'arm_height_difference': ('오른팔을 더 내려주세요.', '오른팔을 더 올려주세요.'),
        'hand_height_difference': ('오른손을 더 내려야 합니다.', '오른손을 더 올려야 합니다.'),
        'direction_difference': ('오른팔 방향을 맞춰주세요.', '오른팔 방향을 맞춰주세요.')
    },
    'left_leg': {
        'bend_angle_difference': ('왼쪽 무릎을 더 굽히세요.', '왼쪽 무릎을 더 펴세요.'),
        'height_difference': ('왼쪽 다리를 더 내리세요.', '왼쪽 다리를 더 올리세요.'),
        'direction_difference': ('왼쪽 다리 방향을 맞춰주세요.', '왼쪽 다리 방향을 맞춰주세요.')
    },
    'right_leg': {
        'bend_angle_difference': ('오른쪽 무릎을 더 굽히세요.', '오른쪽 무릎을 더 펴세요.'),
        'height_difference': ('오른쪽 다리를 더 낮추세요.', '오른쪽 다리를 더 올리세요.'),
        'direction_difference': ('오른쪽 다리 방향을 맞춰주세요.', '오른쪽 다리 방향을 맞춰주세요.')
    },
    'leg':{
        'knee_distance_difference': ('양 무릎을 좀 더 붙여주세요.', '양 무릎을 너무 붙이지 마세요.'),
        'foot_distance_difference': ('양 발을 좀 더 붙여주세요.', '양 발을 너무 붙이지 마세요.')
    }
}

def get_korean_3D_feedback(diffs, angle_thres=20, dist_thres=0.12, height_thres=20):
    feedback = {}

    for body_part, differences in diffs.items():
        feedback[body_part] = {}
        for key, value in differences.items():
            if ('angle' in key or 'direction' in key) and abs(value) < angle_thres:
                continue

            if ('height' in key) and abs(value) < height_thres:
                continue

            if ('distance' in key) and abs(value) < dist_thres:
                continue

            if ('closest' in key):
                continue

            if value < 0:
                feedback[body_part][key] = templates[body_part][key][0]
            elif value > 0:
                feedback[body_part][key] = templates[body_part][key][1]
    
    # 손 상대 위치에 대한 피드백 추가
    left_hand_info = diffs['left_arm']['closest_point_difference']
    right_hand_info = diffs['right_arm']['closest_point_difference']
    if abs(left_hand_info['diff']) > dist_thres and left_hand_info['diff'] < 0:
        feedback['left_arm']['closest_point_difference'] = f"왼손을 좀 더 {body_parts_korean[left_hand_info['target_keypoint']]}에 붙여주세요."
    if abs(right_hand_info['diff']) > dist_thres and right_hand_info['diff'] < 0:
        feedback['right_arm']['closest_point_difference'] = f"오른손을 좀 더 {body_parts_korean[right_hand_info['target_keypoint']]}에 붙여주세요."


    return feedback

def aggregate_feedback(feedback):
    agg_feedback = {}

    ## head feedback
    head = feedback['head']
    if 'lower_angle_difference' in head and 'direction_difference' in head:
        s = head['direction_difference'].replace("세요.", "시고 ") + head['lower_angle_difference']
    elif 'lower_angle_difference' in head:
        s = head['lower_angle_difference']
    elif 'direction_difference' in head:
        s = head['direction_difference']
    else:
        s = ''
    agg_feedback['head'] = s

    ## body feedback
    body = feedback['body']
    if 'bend_angle_difference' in body and 'direction_difference' in body:
        s = body['bend_angle_difference'].replace("세요.", "시고 ") + body['direction_difference'].replace('을', '은')
    elif 'bend_angle_difference' in body:
        s = body['bend_angle_difference']
    elif 'direction_difference' in body:
        s = body['direction_difference']
    else:
        s = ''
    agg_feedback['body'] = s


    ## left_arm feedback
    left_arm = feedback['left_arm']

    # 대략적인 피드백인 direction은 다른 피드백이 존재한다면 일단 제거
    if len(left_arm) > 1:
        # 손의 높이와 팔의 높이에 대한 피드백이 같을 경우 손에 대한 피드백만 남기기
        if 'hand_height_difference' in left_arm and 'arm_height_difference' in left_arm:
            is_hand_down = left_arm['hand_height_difference'] == templates['left_arm']['hand_height_difference'][0]
            is_arm_down = left_arm['arm_height_difference'] == templates['left_arm']['arm_height_difference'][0]

            if (is_hand_down and is_arm_down) or (not is_hand_down and not is_arm_down):
                start_sentence = left_arm['hand_height_difference']
            else:
                start_sentence = random.choice([left_arm['hand_height_difference'], left_arm['arm_height_difference']])
        elif 'hand_height_difference' in left_arm:
            start_sentence = left_arm['hand_height_difference']
        elif 'arm_height_difference' in left_arm:
            start_sentence = left_arm['arm_height_difference']
        else:
            start_sentence = ''
            

        # 만약 굽히는 내용과 높이 피드백이 동시에 존재할 경우 랜덤하게 하나만 선택
        if start_sentence and 'bend_angle_difference' in left_arm:
            start_sentence = random.choice([start_sentence, left_arm['bend_angle_difference']])
        elif 'bend_angle_difference' in left_arm:
            start_sentence = left_arm['bend_angle_difference']
        

        if 'closest_point_difference' in left_arm and 'direction_difference' in left_arm:
            end_sentence = random.choice([left_arm['closest_point_difference'], left_arm['direction_difference']])
        elif 'closest_point_difference' in left_arm:
            end_sentence = left_arm['closest_point_difference']
        elif 'direction_difference' in left_arm:
            end_sentence = left_arm['direction_difference']
        else:
            end_sentence = ''
        
        if start_sentence and end_sentence:
            start_sentence = start_sentence.replace('야 합니다.', '서 ').replace('주세요.', '서 ').replace('히세요.', '혀서 ')
            end_sentence = end_sentence.replace("왼손을 ", "") if "왼손" in start_sentence else end_sentence
            s = start_sentence + end_sentence
        else:
            s = start_sentence if start_sentence else end_sentence

    elif len(left_arm) == 1:
        s = left_arm[list(left_arm)[0]]

    else:
        s = ''
    agg_feedback['left_arm'] = s
    

    ## right_arm feedback
    right_arm = feedback['right_arm']

    # 대략적인 피드백인 direction은 다른 피드백이 존재한다면 일단 제거
    if len(right_arm) > 1:
        # 손의 높이와 팔의 높이에 대한 피드백이 같을 경우 손에 대한 피드백만 남기기
        if 'hand_height_difference' in right_arm and 'arm_height_difference' in right_arm:
            is_hand_down = right_arm['hand_height_difference'] == templates['right_arm']['hand_height_difference'][0]
            is_arm_down = right_arm['arm_height_difference'] == templates['right_arm']['arm_height_difference'][0]

            if (is_hand_down and is_arm_down) or (not is_hand_down and not is_arm_down):
                start_sentence = right_arm['hand_height_difference']
            else:
                start_sentence = random.choice([right_arm['hand_height_difference'], right_arm['arm_height_difference']])
        elif 'hand_height_difference' in right_arm:
            start_sentence = right_arm['hand_height_difference']
        elif 'arm_height_difference' in right_arm:
            start_sentence = right_arm['arm_height_difference']
        else:
            start_sentence = ''
            

        # 만약 굽히는 내용과 높이 피드백이 동시에 존재할 경우 랜덤하게 하나만 선택
        if start_sentence and 'bend_angle_difference' in right_arm:
            start_sentence = random.choice([start_sentence, right_arm['bend_angle_difference']])
        elif 'bend_angle_difference' in right_arm:
            start_sentence = right_arm['bend_angle_difference']
        

        if 'closest_point_difference' in right_arm and 'direction_difference' in right_arm:
            end_sentence = random.choice([right_arm['closest_point_difference'], right_arm['direction_difference']])
        elif 'closest_point_difference' in right_arm:
            end_sentence = right_arm['closest_point_difference']
        elif 'direction_difference' in right_arm:
            end_sentence = right_arm['direction_difference']
        else:
            end_sentence = ''
        
        if start_sentence and end_sentence:
            start_sentence = start_sentence.replace('야 합니다.', '서 ').replace('주세요.', '서 ').replace('히세요.', '혀서 ')
            end_sentence = end_sentence.replace("오른손을 ", "") if "오른손" in start_sentence else end_sentence
            s = start_sentence + end_sentence
        else:
            s = start_sentence if start_sentence else end_sentence

    elif len(right_arm) == 1:
        s = right_arm[list(right_arm)[0]]
        
    else:
        s = ''
    agg_feedback['right_arm'] = s


    ## left leg
    left_leg = feedback['left_leg']
    if 'bend_angle_difference' in left_leg and 'height_difference' in left_leg:
        start_sentence = random.choice([left_leg['height_difference'], left_leg['bend_angle_difference']])
    elif 'bend_angle_difference' in left_leg:
        start_sentence = left_leg['bend_angle_difference']
    elif 'height_difference' in left_leg:
        start_sentence = left_leg['height_difference']
    else:
        start_sentence = ''
    end_sentence = left_leg['direction_difference'] if 'direction_difference' in left_leg else ''

    if start_sentence and end_sentence:
        start_sentence = start_sentence.replace('히세요.', "혀서 ").replace("리세요.", "려서 ")
        s = start_sentence + end_sentence
    elif start_sentence:
        s = start_sentence
    elif end_sentence:
        s = end_sentence
    else:
        s = ''
    agg_feedback['left_leg'] = s


    ## right leg
    right_leg = feedback['right_leg']
    if 'bend_angle_difference' in right_leg and 'height_difference' in right_leg:
        start_sentence = random.choice([right_leg['height_difference'], right_leg['bend_angle_difference']])
    elif 'bend_angle_difference' in right_leg:
        start_sentence = right_leg['bend_angle_difference']
    elif 'height_difference' in right_leg:
        start_sentence = right_leg['height_difference']
    else:
        start_sentence = ''
    end_sentence = right_leg['direction_difference'] if 'direction_difference' in right_leg else ''

    if start_sentence and end_sentence:
        start_sentence = start_sentence.replace('히세요.', "혀서 ").replace("리세요.", "려서 ")
        s = start_sentence + end_sentence
    elif start_sentence:
        s = start_sentence
    elif end_sentence:
        s = end_sentence
    else:
        s = ''
    agg_feedback['right_leg'] = s


    ## leg
    leg = feedback['leg']
    if 'knee_distance_difference' in leg and 'foot_distance_difference' in leg:
        is_knee_close = leg['knee_distance_difference'] == templates['leg']['knee_distance_difference'][0]
        is_foot_close = leg['foot_distance_difference'] == templates['leg']['foot_distance_difference'][0]

        if (is_knee_close and is_foot_close) or (not is_knee_close and not is_foot_close):
            s = leg['foot_distance_difference']
        else:
            s = leg['knee_distance_difference'].replace("을", "은").replace("주세요.", "주시고 ").replace("마세요.", "마시고 ") + leg['foot_distance_difference'].replace("을", "은")
    elif 'knee_distance_difference' in leg:
        s = leg['knee_distance_difference']
    elif 'foot_distance_difference' in leg:
        s = leg['foot_distance_difference']
    else:
        s = ''
    agg_feedback['leg'] = s

    return agg_feedback


def get_connected_sentence_from_dict(agg_feedback):
    startings = [
        "동작 차이를 기반으로 피드백을 드리도록 하겠습니다.",
        "동작 분석을 기반으로 피드백을 제공해 드리겠습니다.",
        "개선점을 안내해 드릴게요.",
        "정확한 동작 분석을 통해 개선할 점을 알려드리겠습니다.",
        "댄스 동작을 비교하여 최적의 피드백을 드릴게요.",
        "자세 차이를 바탕으로 더 나은 동작을 위한 피드백을 드리겠습니다.",
        "댄스 퍼포먼스를 향상시킬 수 있도록 피드백을 시작하겠습니다."
    ]

    good_endings = [
        "나머지 동작는 좋아요! 계속해서 발전해봅시다!",
        "좋은 동작입니다! 앞으로도 꾸준히 연습해볼까요?",
        "완벽에 가까워지고 있어요! 계속 노력해볼게요!",
        "점점 더 좋아지고 있어요! 계속 밀고 나가봅시다!",
        "좋은 흐름이에요! 이대로 쭉 가봅시다!",
        "자세가 많이 발전했어요! 다음 목표를 향해 가볼까요?"
    ]

    bad_endings = [
        "자세가 아직은 많이 좋지 않네요 더 정진해봅시다!",
        "아직은 부족한 부분이 보이지만 꾸준히 연습하면 좋아질 거예요!",
        "자세를 조금 더 신경 쓰면 훨씬 좋아질 거예요! 계속 연습해봐요!",
        "연습을 조금 더 하면 동작이 훨씬 자연스러워질 거예요! 화이팅!",
        "동작의 연결이 자연스럽게 이어질 수 있도록 신경 써보면 좋을 것 같아요!"
    ]

    perfect_msg = [
        "동작 하나하나가 정말 정확하고 힘이 느껴져요!",
        "자연스럽고 부드러운 흐름이 너무 멋져요!",
        "디테일이 살아 있어요!",
        "춤선이 정말 아름답고 완벽하게 살아 있네요!",
        "스텝이 너무 정확하고 깔끔해서 감탄했어요!"
    ]

    start_word = [
        "우선, ",
    ]

    connect_word = [
        "계속해서 ",
        "다음으로는 ",
        "이어서 ",
        "그리고 ",
        "그 다음엔, ",
    ]

    if len(agg_feedback) >= 5:
        ending = random.choice(bad_endings)
    else:
        ending = random.choice(good_endings)


    feedback_string = ''
    idx = 0
    for k, feedback in agg_feedback.items():
        if not feedback:
            continue
        
        if idx == 0:
            feedback_string += random.choice(start_word)
            idx += 1
        else:
            feedback_string += random.choice(connect_word)

        feedback_string += (feedback + '\n')
    
    if not feedback_string:
        return random.choice(perfect_msg)
    else:
        feedback_string = random.choice(startings) + "\n\n" + feedback_string + ending
        return feedback_string
    

if __name__ == "__main__":
    img_path_1 = './images/jun_ude.jpg'
    img_path_2 = './images/jun_v.jpg'


    import sys
    sys.path.append("./")
    from dance_scoring import detector, scoring
    from data_pipeline.pipeline import refine_float_dict
    import config
    from dance_feedback.pose_compare import extract_3D_pose_landmarks


    # 랜드마크 추출
    det = detector.PoseDetector()
    landmark_1, _, _, _ = det.get_image_landmarks(img_path_1)
    landmark_2, _, _, _ = det.get_image_landmarks(img_path_2)
    

    # 정규화하기
    pose_landmarks_np_1 = scoring.refine_landmarks(landmark_1)
    pose_landmarks_np_2 = scoring.refine_landmarks(landmark_2)
    normalized_pose_landmarks_np_2 = scoring.normalize_landmarks_to_range(
        scoring.refine_landmarks(landmark_1, target_keys=config.TOTAL_KEYPOINTS), 
        scoring.refine_landmarks(landmark_2, target_keys=config.TOTAL_KEYPOINTS)
    )
    for i, landmarks in enumerate(landmark_2):
        landmarks.x = normalized_pose_landmarks_np_2[i, 0]
        landmarks.y = normalized_pose_landmarks_np_2[i, 1]
        landmarks.z = normalized_pose_landmarks_np_2[i, 2]


    # difference를 기반으로 결과 뽑기
    result_1 = extract_3D_pose_landmarks(landmark_1)
    result_2 = extract_3D_pose_landmarks(landmark_2)
    diffs = get_difference_dict(result_1, result_2)
    feedback = get_korean_3D_feedback(diffs)
    agg_feedback = aggregate_feedback(feedback)

    print(refine_float_dict(diffs))