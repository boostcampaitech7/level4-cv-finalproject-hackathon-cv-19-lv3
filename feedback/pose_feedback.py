import math
import json
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_feedback(feature_differences, threshold=30):
    """
    Generate feedback based on the feature differences provided.

    Parameters:
    - feature_differences (dict): Dictionary of feature differences.
    - threshold (int): Minimum difference required to generate feedback (default: 30).

    Returns:
    - dict: Feedback for each feature exceeding the threshold.
    """
    feedback_dict = {}  # Store feedback for each feature

    # Iterate through the features and generate feedback if the threshold is exceeded
    for feature, difference in feature_differences.items():
        if abs(difference) >= threshold:
            if feature == "head_difference":
                feedback_dict["head"] = "Tilt your head to the left." if difference > 0 else "Tilt your head to the right."

            elif feature == "shoulder_difference":
                if difference > 0:
                    random_choice = random.choice(["Lower your left shoulder.", "Raise your right shoulder."])
                else:
                    random_choice = random.choice(["Raise your left shoulder.", "Lower your right shoulder."])
                feedback_dict["shoulder"] = random_choice

            elif feature in ["left_arm_angle_difference", "right_arm_angle_difference"]:
                side = "left" if "left" in feature else "right"
                if difference > 0:
                    feedback_dict[feature.replace("_angle_difference", "")] = f"Rotate your {side} arm clockwise."
                else:
                    feedback_dict[feature.replace("_angle_difference", "")] = f"Rotate your {side} arm counterclockwise."

            elif feature in ["left_elbow_angle_difference", "right_elbow_angle_difference"]:
                side = "left" if "left" in feature else "right"
                feedback_dict[feature.replace("_angle_difference", "")] = f"Straighten your {side} elbow." if difference > 0 else f"Bend your {side} elbow."

            elif feature in ["left_leg_angle_difference", "right_leg_angle_difference"]:
                side = "left" if "left" in feature else "right"
                if difference > 0:
                    feedback_dict[feature.replace("_angle_difference", "")] = f"Rotate your {side} leg clockwise."
                else:
                    feedback_dict[feature.replace("_angle_difference", "")] = f"Rotate your {side} leg counterclockwise."

            elif feature in ["left_knee_angle_difference", "right_knee_angle_difference"]:
                side = "left" if "left" in feature else "right"
                feedback_dict[feature.replace("_angle_difference", "")] = f"Straighten your {side} knee." if difference > 0 else f"Bend your {side} knee."

    # If no features exceed the threshold, return a default success message
    if not feedback_dict:
        return {"perfect_msg": "Great job! Your posture is perfect!"}

    return feedback_dict

def generate_korean_feedback(feature_differences, threshold = 30):
    """
    주어진 특징 차이에 따라 피드백을 생성합니다.

    Parameters:
    - feature_differences (dict): 특징 차이를 저장한 딕셔너리.
    - threshold (int): 피드백을 생성할 최소 차이값 기준 (기본값: 30).

    Returns:
    - dict: 기준값을 초과한 각 특징에 대한 피드백.
    """
    feedback_dict = {}  # 각 특징에 대한 피드백 저장

    # 특징을 반복하며 기준값을 초과하는 경우 피드백 생성
    for feature, difference in feature_differences.items():
        if abs(difference) >= threshold:
            modifier = "더" if abs(difference) > threshold*2 else "조금 더"

            if feature == "head difference":
                feedback_dict["head"] = f"머리를 {modifier} 왼쪽으로 기울여주세요." if difference > 0 else f"머리를 {modifier} 오른쪽으로 기울여주세요."

            elif feature == "shoulder difference":
                modifier = "너무" if abs(difference) > threshold*2 else "약간"
                if difference > 0:
                    feedback_dict["shoulder"] = f"왼쪽 어깨를 조금 내려주세요."
                else:
                    feedback_dict["shoulder"] = f"오른쪽 어깨를 조금 내려주세요."

            elif feature in ["left arm angle difference", "right arm angle difference"]:
                modifier = "전혀" if abs(difference) > threshold*2 else "약간"
                side = "왼쪽" if "left" in feature else "오른쪽"
                to = "오른쪽" if difference > 0 else "왼쪽"
                feedback_dict[feature.replace(" angle difference", "")] = f"{side} 팔을 좀 더 {to}으로 움직여주세요."
            
            elif feature in ["left elbow angle difference", "right elbow angle difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace(" angle difference", "")] = f"{side} 팔꿈치를 {modifier} 펴주세요." if difference > 0 else f"{side} 팔꿈치를 {modifier} 구부려주세요."

            elif feature in ["left leg angle difference", "right leg angle difference"]:
                modifier = "전혀" if abs(difference) > threshold*2 else "약간"
                side = "왼쪽" if "left" in feature else "오른쪽"
                to = "오른쪽" if difference > 0 else "왼쪽"
                feedback_dict[feature.replace(" angle difference", "")] = f"{side} 다리를 좀 더 {to}으로 움직여주세요."
                    
            elif feature in ["left knee angle difference", "right knee angle difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace(" angle difference", "")] = f"{side} 무릎을 {modifier} 펴주세요." if difference > 0 else f"{side} 무릎을 {modifier} 구부려주세요."
    
    # 기준값을 초과한 특징이 없으면 기본 성공 메시지 반환
    if not feedback_dict:
        perfect_msg = [
            "동작 하나하나가 정말 정확하고 힘이 느껴져요!",
            "자연스럽고 부드러운 흐름이 너무 멋져요!",
            "디테일이 살아 있어요!",
            "춤선이 정말 아름답고 완벽하게 살아 있네요!",
            "스텝이 너무 정확하고 깔끔해서 감탄했어요!"
        ]
        return {"perfect msg": random.choice(perfect_msg)}
    
    return feedback_dict


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

class frame_pose:
    def __init__(self, landmarks_data):
        self.left_ear = np.array([
            landmarks_data['head']['7']['x'],
            landmarks_data['head']['7']['y']
        ])
        self.right_ear = np.array([
            landmarks_data['head']['8']['x'],
            landmarks_data['head']['8']['y']
        ])
        self.left_shoulder = np.array([
            landmarks_data['left_arm']['11']['x'],
            landmarks_data['left_arm']['11']['y']
        ])
        self.right_shoulder = np.array([
            landmarks_data['right_arm']['12']['x'],
            landmarks_data['right_arm']['12']['y']
        ])
        self.left_elbow = np.array([
            landmarks_data['left_arm']['13']['x'],
            landmarks_data['left_arm']['13']['y']
        ])
        self.right_elbow = np.array([
            landmarks_data['right_arm']['14']['x'],
            landmarks_data['right_arm']['14']['y']
        ])
        self.left_wrist = np.array([
            landmarks_data['left_arm']['15']['x'],
            landmarks_data['left_arm']['15']['y']
        ])
        self.right_wrist = np.array([
            landmarks_data['right_arm']['16']['x'],
            landmarks_data['right_arm']['16']['y']
        ])
        self.left_pelvis = np.array([
            landmarks_data['left_leg']['23']['x'],
            landmarks_data['left_leg']['23']['y']
        ])
        self.right_pelvis = np.array([
            landmarks_data['right_leg']['24']['x'],
            landmarks_data['right_leg']['24']['y']
        ])
        self.left_knee = np.array([
            landmarks_data['left_leg']['25']['x'],
            landmarks_data['left_leg']['25']['y']
        ])
        self.right_knee = np.array([
            landmarks_data['right_leg']['26']['x'],
            landmarks_data['right_leg']['26']['y']
        ])
        self.left_ankle = np.array([
            landmarks_data['left_leg']['27']['x'],
            landmarks_data['left_leg']['27']['y'],
        ])
        self.right_ankle = np.array([
            landmarks_data['right_leg']['28']['x'],
            landmarks_data['right_leg']['28']['y']
        ])

    def get_ear_height_difference(self):
        return calculate_two_points_angle(self.right_ear, self.left_ear)
    
    def get_shoulder_height_difference(self):
        return  calculate_two_points_angle(self.right_shoulder, self.left_shoulder)

    def get_left_arm_angle(self):
        return calculate_two_points_angle(self.left_shoulder, self.left_wrist)

    def get_right_arm_angle(self):
        return calculate_two_points_angle(self.right_shoulder, self.right_wrist)

    def get_left_elbow_angle(self):
        return calculate_three_points_angle(self.left_shoulder, self.left_elbow, self.left_wrist)

    def get_right_elbow_angle(self):
        return calculate_three_points_angle(self.right_shoulder, self.right_elbow, self.right_wrist)

    def get_left_leg_angle(self):
        return calculate_two_points_angle(self.left_pelvis, self.left_knee)

    def get_right_leg_angle(self):
        return calculate_two_points_angle(self.right_pelvis, self.right_knee)

    def get_left_knee_angle(self):
        return calculate_three_points_angle(self.left_pelvis, self.left_knee, self.left_ankle)

    def get_right_knee_angle(self):
        return calculate_three_points_angle(self.right_pelvis, self.right_knee, self.right_ankle)

def get_point(data, key1, key2):
    return np.array([data[key1][key2]['x'], data[key1][key2]['y']])


def json_to_prompt(target_landmarks_json_path, compare_landmarks_json_path):
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

    pose1 = frame_pose(data1)
    pose2 = frame_pose(data2)
    
    # result_json 생성 및 저장
    result_json = {
        "head difference": int(pose1.get_ear_height_difference() - pose2.get_ear_height_difference()),
        "shoulder difference": int(pose1.get_shoulder_height_difference() - pose2.get_shoulder_height_difference()),
        "left arm angle difference": int(abs(pose1.get_left_arm_angle()) - abs(pose2.get_left_arm_angle())),
        "right arm angle difference": int(abs(pose1.get_right_arm_angle()) - abs(pose2.get_right_arm_angle())),
        "left elbow angle difference": int(pose1.get_left_elbow_angle() - pose2.get_left_elbow_angle()),
        "right elbow angle difference": int(pose1.get_right_elbow_angle() - pose2.get_right_elbow_angle()),
        "left leg angle difference": int(abs(pose1.get_left_leg_angle()) - abs(pose2.get_left_leg_angle())),
        "right leg angle difference": int(abs(pose1.get_right_leg_angle()) - abs(pose2.get_right_leg_angle())),
        "left knee angle difference": int(pose1.get_left_knee_angle() - pose2.get_left_knee_angle()),
        "right knee angle difference": int(pose1.get_right_knee_angle() - pose2.get_right_knee_angle()),
    }
    return result_json


###################### NEW x, y, z, dataset test
def calculate_two_vector_angle(v1, v2, normal, eps=1e-7):
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
    vector = point2 - point1
    angle = math.degrees(math.atan2(vector[0], vector[1]))

    return angle

def calculate_three_points_angle_with_sign(point1, point2, point3, normal, eps=1e-7):
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


def visualize_angle(p1, p2, p3):
    # 벡터 정의
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # 단위 벡터 계산
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    
    # 각도 계산 (라디안)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # 플로팅
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--')
    
    # 점 그리기
    ax.scatter(*p1, color='red', label='P1')
    ax.scatter(*p2, color='blue', label='P2 (Center)')
    ax.scatter(*p3, color='green', label='P3')
    
    # 벡터 그리기
    ax.quiver(*p2, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(*p2, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green')
    
    # 각도 표시
    ax.text(p2[0], p2[1], f'{angle_deg:.2f}°', fontsize=12, ha='right', color='black')
    
    ax.legend()
    plt.show()
    
    return angle_deg

def visualize_vector(v1, v2):
    # 단위 벡터 계산
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    
    # 각도 계산 (라디안)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # 플로팅
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--')
    
    # 원점 및 벡터 그리기
    origin = np.array([0, 0])
    ax.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label='V1')
    ax.quiver(*origin, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green', label='V2')
    
    ax.legend()
    plt.show()
    
    return angle_deg


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


def generate_3D_feedback(result, threshold=30):
    feedback = {}
    
    # 피드백 문구 템플릿
    templates = {
        'head': {
            'lower_angle_difference': ('고개를 좀 더 숙이세요.', '고개를 좀 더 들어 올리세요.'),
            'tilt_angle_difference': ('고개를 좀 더 오른쪽으로 꺾으세요.', '고개를 좀 더 왼쪽으로 꺾으세요.'),
            'direction_difference': ('시선을 더 오른쪽으로 돌리세요.', '시선을 더 왼쪽으로 돌리세요.')
        },
        'waist': {
            'bend_angle_difference': ('허리를 더 굽히세요.', '허리를 더 펴세요.'),
            'direction_difference': ('몸을 더 오른쪽으로 돌리세요.', '몸을 더 왼쪽으로 돌리세요.')
        },
        'left_arm': {
            'bend_angle_difference': ('왼팔을 더 굽히세요.', '왼팔을 더 펴세요.'),
            'height_difference': ('왼팔을 더 내려주세요.', '왼팔을 더 올려주세요.'),
            'direction_difference': ('왼팔 방향을 조정하세요.', '왼팔 방향을 조정하세요.')
        },
        'right_arm': {
            'bend_angle_difference': ('오른팔을 더 굽히세요.', '오른팔을 더 펴세요.'),
            'height_difference': ('오른팔을 더 내려주세요.', '오른팔을 더 올려주세요.'),
            'direction_difference': ('오른팔 방향을 조정하세요.', '오른팔 방향을 조정하세요.')
        },
        'left_leg': {
            'bend_angle_difference': ('왼쪽 무릎을 더 굽히세요.', '왼쪽 무릎을 더 펴세요.'),
            'height_difference': ('왼다리를 더 낮추세요.', '왼다리를 더 올리세요.'),
            'direction_difference': ('왼다리 방향을 조정하세요.', '왼다리 방향을 조정하세요.')
        },
        'right_leg': {
            'bend_angle_difference': ('오른쪽 무릎을 더 굽히세요.', '오른쪽 무릎을 더 펴세요.'),
            'height_difference': ('오른다리를 더 낮추세요.', '오른다리를 더 올리세요.'),
            'direction_difference': ('오른다리 방향을 조정하세요.', '오른다리 방향을 조정하세요.')
        }
    }
    
    for body_part, differences in result.items():
        messages = []
        for key, value in differences.items():
            if value < threshold:
                continue

            if value < 0:
                messages.append(templates[body_part][key][0])
            elif value > 0:
                messages.append(templates[body_part][key][1])
        
        if messages:
            feedback[body_part] = ' '.join(messages)
    
    return feedback


class FramePose3D:
    def __init__(self, landmarks_data):
        self.nose = np.array([
            landmarks_data['head']['0']['x'],
            landmarks_data['head']['0']['y'],
            landmarks_data['head']['0']['z']
        ])

        self.left_eye = np.array([
            landmarks_data['head']['2']['x'],
            landmarks_data['head']['2']['y'],
            landmarks_data['head']['2']['z']
        ])

        self.right_eye = np.array([
            landmarks_data['head']['5']['x'],
            landmarks_data['head']['5']['y'],
            landmarks_data['head']['5']['z']
        ])

        self.left_ear = np.array([
            landmarks_data['head']['7']['x'],
            landmarks_data['head']['7']['y'],
            landmarks_data['head']['7']['z']
        ])
        self.right_ear = np.array([
            landmarks_data['head']['8']['x'],
            landmarks_data['head']['8']['y'],
            landmarks_data['head']['8']['z']
        ])
        self.left_mouth = np.array([
            landmarks_data['head']['9']['x'],
            landmarks_data['head']['9']['y'],
            landmarks_data['head']['9']['z']
        ])
        self.right_mouth = np.array([
            landmarks_data['head']['10']['x'],
            landmarks_data['head']['10']['y'],
            landmarks_data['head']['10']['z']
        ])

        self.left_shoulder = np.array([
            landmarks_data['left_arm']['11']['x'],
            landmarks_data['left_arm']['11']['y'],
            landmarks_data['left_arm']['11']['z']
        ])
        self.right_shoulder = np.array([
            landmarks_data['right_arm']['12']['x'],
            landmarks_data['right_arm']['12']['y'],
            landmarks_data['right_arm']['12']['z']
        ])
        self.left_elbow = np.array([
            landmarks_data['left_arm']['13']['x'],
            landmarks_data['left_arm']['13']['y'],
            landmarks_data['left_arm']['13']['z']
        ])
        self.right_elbow = np.array([
            landmarks_data['right_arm']['14']['x'],
            landmarks_data['right_arm']['14']['y'],
            landmarks_data['right_arm']['14']['z']
        ])
        self.left_wrist = np.array([
            landmarks_data['left_arm']['15']['x'],
            landmarks_data['left_arm']['15']['y'],
            landmarks_data['left_arm']['15']['z']
        ])
        self.right_wrist = np.array([
            landmarks_data['right_arm']['16']['x'],
            landmarks_data['right_arm']['16']['y'],
            landmarks_data['right_arm']['16']['z']
        ])
        self.left_pelvis = np.array([
            landmarks_data['left_leg']['23']['x'],
            landmarks_data['left_leg']['23']['y'],
            landmarks_data['left_leg']['23']['z']
        ])
        self.right_pelvis = np.array([
            landmarks_data['right_leg']['24']['x'],
            landmarks_data['right_leg']['24']['y'],
            landmarks_data['right_leg']['24']['z']
        ])
        self.left_knee = np.array([
            landmarks_data['left_leg']['25']['x'],
            landmarks_data['left_leg']['25']['y'],
            landmarks_data['left_leg']['25']['z']
        ])
        self.right_knee = np.array([
            landmarks_data['right_leg']['26']['x'],
            landmarks_data['right_leg']['26']['y'],
            landmarks_data['right_leg']['26']['z']
        ])
        self.left_ankle = np.array([
            landmarks_data['left_leg']['27']['x'],
            landmarks_data['left_leg']['27']['y'],
            landmarks_data['left_leg']['27']['z']
        ])
        self.right_ankle = np.array([
            landmarks_data['right_leg']['28']['x'],
            landmarks_data['right_leg']['28']['y'],
            landmarks_data['right_leg']['28']['z']
        ])

        # 몸이 움직임에 따라 좌표축 역할을 해줄 2개의 벡터 정의
        x_direction = (self.left_shoulder - self.right_shoulder)
        x_direction[1] = 0.
        self.shoulder_vector = x_direction # 팔과 머리의 움직임과 연관

        x_direction = (self.left_pelvis - self.right_pelvis)
        x_direction[1] = 0.
        self.pelvis_vector = x_direction # 다리의 움직임과 연관

    #################################### HEAD
    def get_head_angle_1(self):
        '''
        귀 중점, 어깨중점, 엉덩이 중점의 3point
        - 고개가 앞/뒤로 얼마나 숙여져/젖혀져 있는지 구합니다.
        - 0 ~ 180 : 앞으로 숙임
        - 0 ~ -180 : 뒤로 젖힘
        '''
        shoulder_center = (self.right_shoulder + self.left_shoulder) / 2
        hip_center = (self.left_pelvis + self.right_pelvis) / 2
        ear_center = (self.right_ear + self.left_ear) / 2
        return calculate_three_points_angle(hip_center, shoulder_center, ear_center)
    
    def get_head_angle_2(self):
        '''
        - 고개가 어깨를 기준으로 어느쪽으로 얼마나 꺾여있는지 구합니다.
        - 0인경우 안꺾임
        - 왼쪽으로 꺾여 있으면 양수
        - 오른쪽으로 꺾여 있으면 음수
        '''
        ear_vector = (self.left_ear - self.right_ear)
        return calculate_two_vector_angle(self.shoulder_vector, ear_vector, normal=np.array([0, 0, 1]))
    
    def get_eye_direction(self):
        '''
        - 시선이 카메라를 기준으로 왼쪽/오른쪽 어디를 바라보고 있는지 구합니다
        - 180이 정면, 0이 완전 후면
        - 음수면 오른쪽 보는중, 양수면 왼쪽 보는중
        '''
        eye_center = (self.right_eye + self.left_eye) / 2
        ear_center = (self.right_ear + self.left_ear) / 2

        return calculate_two_points_angle_reverse(ear_center[[0, 2]], eye_center[[0, 2]])

    #################################### BODY
    def get_waist_angle_1(self):
        '''
        어깨중점, 엉덩이중점, 무릎 중점의 3 point
        - 허리 굽혀진 정도 계산
        - 180에 가까울수록 펴져있고, 0에 가까울수록 접혀져있음
        '''
        shoulder_center = (self.right_shoulder + self.left_shoulder) / 2
        hip_center = (self.left_pelvis + self.right_pelvis) / 2
        knee_center = (self.left_knee + self.right_knee) / 2
        return calculate_three_points_angle(shoulder_center, hip_center, knee_center)
    
    def get_waist_angle_2(self):
        '''
        - 사람의 몸이 전체적으로 얼마나 뒤를 돌아 있는지 판단하기위해 엉덩이 point 2개를 이용
        - 왼쪽으로 돌고 있으면 양수, 오른쪽으로 돌고 있으면 음수
        '''
        return calculate_two_points_angle(self.right_pelvis[[0, 2]], self.left_pelvis[[0, 2]])
    
    #################################### LEFT ARM
    def get_left_elbow_angle(self):
        # 왼팔 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(self.left_shoulder, self.left_elbow, self.left_wrist)
    
    def get_left_arm_height(self):
        '''
        - 왼팔이 얼마나 올라가 있는지
        - 팔이 높게 위치할수록 1에 가깝고, 낮게 위치할수록 0
        '''
        hip_to_shoulder = self.left_pelvis[1] - self.left_shoulder[1]
        y_max = self.left_pelvis[1]
        y_min = self.left_shoulder[1] - hip_to_shoulder

        value = (np.clip(self.left_wrist[1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)
    
    def get_left_arm_dir(self):
        '''
        왼팔의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.left_shoulder[[0, 2]], self.left_elbow[[0, 2]])

    #################################### RIGHT ARM
    def get_right_elbow_angle(self):
        # 오른팔 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(self.right_shoulder, self.right_elbow, self.right_wrist)
    

    def get_right_arm_height(self):
        '''
        - 오른팔이 얼마나 올라가 있는지
        - 팔이 높게 위치할수록 1에 가깝고, 낮게 위치할수록 0
        '''
        hip_to_shoulder = self.right_pelvis[1] - self.right_shoulder[1]
        y_max = self.right_pelvis[1]
        y_min = self.right_shoulder[1] - hip_to_shoulder

        value = (np.clip(self.right_wrist[1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)

    def get_right_arm_dir(self):
        '''
        오른팔의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.right_shoulder[[0, 2]], self.right_elbow[[0, 2]])

    #################################### LEFT LEG
    def get_left_knee_angle(self):
        # 왼다리 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(self.left_pelvis, self.left_knee, self.left_ankle)
    
    def get_left_leg_height(self):
        '''
        - 골반을 기준으로 왼다리를 얼마나 올라가 있는지.
        - 아래에 위치할수록 0, 위로갈수록 1
        '''
        hip_to_ground = 0.9999 - self.left_pelvis[1]
        y_max = 0.9999
        y_min = self.left_pelvis[1] - hip_to_ground

        value = (np.clip(self.left_ankle[1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)
    
    def get_left_leg_dir(self):
        '''
        왼다리의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.left_pelvis[[0, 2]], self.left_knee[[0, 2]])

    #################################### RIGHT LEG
    def get_right_knee_angle(self):
        # 오른다리 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(self.right_pelvis, self.right_knee, self.right_ankle)
    
    def get_right_leg_height(self):
        '''
        - 골반을 기준으로 오른다리가 얼마나 올라가 있는지.
        - 아래에 위치할수록 0, 위로갈수록 1
        '''
        hip_to_ground = 0.9999 - self.right_pelvis[1]
        y_max = 0.9999
        y_min = self.right_pelvis[1] - hip_to_ground

        value = (np.clip(self.right_ankle[1], y_min, y_max) - y_min) / (y_max - y_min)
        return (1 - value)

    
    def get_right_leg_dir(self):
        '''
        오른다리의 방향각을 구합니다
        - 0 ~ 180 몸 뒤쪽
        - 0 ~ -180 몸 앞쪽
        '''
        return calculate_two_points_angle(self.right_pelvis[[0, 2]], self.right_knee[[0, 2]])
    
    #################################### EXTRA
    def get_leg_angle(self):
        '''
        다리가 벌려져 있을수록 180, 모아져 있을수록 0
        '''
        hip_center = (self.left_pelvis + self.right_pelvis) / 2
        return calculate_three_points_angle(
            self.left_knee, hip_center, self.right_knee
        )


def json_to_prompt_2(target_landmarks_json_path, compare_landmarks_json_path):
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
    result = {
        'head':{
            'lower_angle_difference': int(pose1.get_head_angle_1() - pose2.get_head_angle_1()), # 음수인 경우 고개를 좀 더 숙여라, 양수인 경우 고개를 좀 더 들어라
            'tilt_angle_difference': int(pose1.get_head_angle_2() - pose2.get_head_angle_2()), # 음수인 경우 고개를 좀 더 오른쪽으로 꺾어라, 양수인 경우 고개를 좀 더 왼쪽으로 꺾어라
            'direction_difference': int(pose1.get_eye_direction() - pose2.get_eye_direction()) # 음수인 경우 좀 더 오른쪽을 바라봐라, 양수인 경우 좀 더 왼쪽을 바라봐라
        },
        'waist':{
            'bend_angle_difference': int(pose1.get_waist_angle_1() - pose2.get_waist_angle_1()), # 음수면 허리를 더 굽혀라, 양수면 허리를 더 펴라
            'direction_difference': int(pose1.get_waist_angle_2() - pose2.get_waist_angle_2()) # 음수면 몸을 더 오른쪽으로 돌려라, 양수면 몸을 더 왼쪽으로 돌려라
        },
        'left_arm':{
            'bend_angle_difference': int(pose1.get_left_elbow_angle() - pose2.get_left_elbow_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'height_difference': int(180 * (pose1.get_left_arm_height() - pose2.get_left_arm_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'direction_difference': int(pose1.get_left_arm_dir() - pose2.get_left_arm_dir()) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
        },
        'right_arm':{
            'bend_angle_difference': int(pose1.get_right_elbow_angle() - pose2.get_right_elbow_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'height_difference': int(180 * (pose1.get_right_arm_height() - pose2.get_right_arm_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'direction_difference': int(pose1.get_right_arm_dir() - pose2.get_right_arm_dir()) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
        },
        'left_leg':{
            'bend_angle_difference': int(pose1.get_left_knee_angle() - pose2.get_left_knee_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'height_difference': int(180 * (pose1.get_left_leg_height() - pose2.get_left_leg_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'direction_difference': int(pose1.get_left_leg_dir() - pose2.get_left_leg_dir()) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
        },
        'right_leg':{
            'bend_angle_difference': int(pose1.get_right_knee_angle() - pose2.get_right_knee_angle()), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
            'height_difference': int(180 * (pose1.get_right_leg_height() - pose2.get_right_leg_height())), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
            'direction_difference': int(pose1.get_right_leg_dir() - pose2.get_right_leg_dir()) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
        }
    }
    return result


######################################################################################## POSE SCRIPT 시도해보기
class FramePoseScript:
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
        self.keypoints['neck'] = (self.keypoints['left_mouth'] + self.keypoints['right_mouth']) / 2
        self.keypoints['pelvis'] = (self.keypoints['left_hip'] + self.keypoints['right_hip']) / 2
        self.keypoints['torso'] = (self.keypoints['left_shoulder'] + self.keypoints['right_shoulder'] + self.keypoints['left_hip'] + self.keypoints['right_hip']) / 4
        self.keypoints['left_hand'] = (self.keypoints['left_pinky'] + self.keypoints['left_index'] + self.keypoints['left_thumb']) / 3
        self.keypoints['right_hand'] = (self.keypoints['right_pinky'] + self.keypoints['right_index'] + self.keypoints['right_thumb']) / 3

        self.ALL_DISTANCE_PAIRS = [
            ("left_elbow", "right_elbow"), # 두 팔꿈치를 붙여주세요
            ("left_hand", "right_hand"), # 두 손을 더 모아주세요
            ("left_knee", "right_knee"), # 두 무릎을 붙여주세요
            ("left_foot", "right_foot"), # 두 발을 더 모아주세요
            ("left_hand", "left_shoulder"), 
            ("left_hand", "right_shoulder"),
            ("right_hand", "left_shoulder"),
            ("right_hand", "right_shoulder"),
            ("left_hand", "right_elbow"),
            ("right_hand", "left_elbow"),
            ("left_hand", "left_knee"),
            ("left_hand", "right_knee"),
            ("right_hand", "left_knee"),
            ("right_hand", "right_knee"),
            ("left_hand", "left_ankle"),
            ("left_hand", "right_ankle"),
            ("right_hand", "left_ankle"),
            ("right_hand", "right_ankle"),
            ("left_hand", "left_foot"),
            ("left_hand", "right_foot"),
            ("right_hand", "left_foot"),
            ("right_hand", "right_foot"),
        ]

        self.ALL_REL_PAIRS = [ 
            ("left_shoulder", "right_shoulder"),
            # ("left_hand", "right_hand"),
            # ("left_foot", "right_foot"),
            # ("left_hip", "left_knee"),
            # ("right_hip", "right_knee"),
            # ("left_hand", "left_shoulder"),
            # ("right_hand", "right_shoulder"),
            # ("left_foot", "left_hip"),
            # ("right_foot", "right_hip"),
            # ("left_hand", "left_hip"),
            # ("right_hand", "right_hip"),
            ("left_hand", "torso"),
            ("right_hand", "torso"),
            ("left_foot", "torso"),
            ("right_foot", "torso"),
        ]

        self.REL_PAIR_FILTERING = [
            ('y', 'z'),
            # ('x', 'y', 'z'),
            # ('x', 'y', 'z'),
            # ('y'),
            # ('y'),
            # ('x', 'y'),
            # ('x', 'y'),
            # ('x', 'y'),
            # ('x', 'y'),
            # ('y'),
            # ('y'),
            ('z'),
            ('z'),
            ('z'),
            ('z'),
        ]
    
    ########################################### ANGLE FEATURES -> 차이값을 통해 양수면 더 펴야됨, 음수면 더 굽혀야됨
    def get_left_knee_angle(self):
        # 왼다리 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(
            self.keypoints["left_hip"], 
            self.keypoints["left_knee"], 
            self.keypoints["left_ankle"]
        )

    def get_right_knee_angle(self):
        # 오른다리 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(
            self.keypoints["right_hip"], 
            self.keypoints["right_knee"], 
            self.keypoints["right_ankle"]
        )

    def get_left_elbow_angle(self):
        # 왼팔 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(
            self.keypoints["left_shoulder"], 
            self.keypoints["left_elbow"], 
            self.keypoints["left_wrist"]
        )

    def get_right_elbow_angle(self):
        # 오른팔 굽혀진 정도. 0 ~ 180
        return calculate_three_points_angle(
            self.keypoints["right_shoulder"], 
            self.keypoints["right_elbow"], 
            self.keypoints["right_wrist"]
        )
    
    def get_all_angle_features(self):
        return {
            "left_knee_angle": self.get_left_knee_angle(),
            "right_knee_angle": self.get_right_knee_angle(),
            "left_elbow_angle": self.get_left_elbow_angle(),
            "right_elbow_angle": self.get_right_elbow_angle()
        }
    
    ########################################### DISTANCE FEATURES
    # -> 임계값을 사용해서 
    def compute_distance(self, point1, point2):
        """ 두 점 사이의 L2 거리(유클리드 거리) 계산 """
        return np.linalg.norm(self.keypoints[point1] - self.keypoints[point2])

    def get_all_distance_features(self):
        return {
            f"{f} {t} distance": self.compute_distance(f, t) for (f, t) in self.ALL_DISTANCE_PAIRS
        }
    
    ########################################### RELATIVE POSITIONS
    def compute_rel_position(self, point1, point2, idx):
        """
        Compute the relative position of point1 with respect to point2 along X, Y, and Z axes.
        
        Args:
            point1 (str): Key name of the first point in self.keypoints.
            point2 (str): Key name of the second point in self.keypoints.

        Returns:
            dict: A dictionary with relative positions along X, Y, and Z axes.
        """
        p1 = self.keypoints[point1]
        p2 = self.keypoints[point2]

        # X-axis comparison
        diff_x = p1[0] - p2[0]
        # Y-axis comparison
        diff_y = p1[1] - p2[1]
        # Z-axis comparison
        diff_z = p1[2] - p2[2]
        return {
            k: v  for k, v in {'x': diff_x, 'y': diff_y, 'z': diff_z}.items() if k in self.REL_PAIR_FILTERING[idx]
        }
    
    def get_all_rel_position(self):
        return {
            f"{f} {t} rel": self.compute_rel_position(f, t, idx) for idx, (f, t) in enumerate(self.ALL_REL_PAIRS)
        }

def json_to_prompt_3(target_landmarks_json_path, compare_landmarks_json_path):
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

    pose1 = FramePoseScript(data1)
    pose2 = FramePoseScript(data2)

    # 각도 관련 차이
    angle_features_1 = pose1.get_all_angle_features()
    angle_features_2 = pose2.get_all_angle_features()
    angle_difference = {
        k: angle_features_1[k] - angle_features_2[k] for k in angle_features_1.keys()
    }

    # 거리 관련 차이
    dist_features_1 = pose1.get_all_distance_features()
    dist_features_2 = pose2.get_all_distance_features()
    dist_difference = {
        k: dist_features_1[k] - dist_features_2[k] for k in dist_features_1.keys()
    }

    # 상대 위치 차이
    rel_pos_1 = pose1.get_all_rel_position()
    rel_pos_2 = pose2.get_all_rel_position()
    rel_pos_difference = {}
    for k in rel_pos_1.keys():
        tmp = {}
        for axis in rel_pos_1[k].keys():
            tmp[axis] = {
                'diff': rel_pos_1[k][axis] - rel_pos_2[k][axis],
                'pose1': rel_pos_1[k][axis],
                'pose2': rel_pos_2[k][axis]
            }
        rel_pos_difference[k] = tmp

    return {
        'angle_difference': angle_difference,
        'distance_difference': dist_difference,
        'relative_pos_difference': rel_pos_difference
    }

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
    "left_hip": "왼쪽 엉덩이",
    "right_hip": "오른쪽 엉덩이",
    "left_knee": "왼쪽 무릎",
    "right_knee": "오른쪽 무릎",
    "left_ankle": "왼쪽 발목",
    "right_ankle": "오른쪽 발목",
    "left_heel": "왼쪽 발뒤꿈치",
    "right_heel": "오른쪽 발뒤꿈치",
    "left_foot_index": "왼쪽 발가락",
    "right_foot_index": "오른쪽 발가락",
    # 방향을 고려하지 않은 일반적인 번역 추가
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
    "torso": "몸"
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


def get_korean_feedback_posescript(diffs, angle_thres=20, distance_thres=0.15, rel_thres=0.15):
    feedback = []

    # angle feedback 생성
    for k, v in diffs['angle_difference'].items():
        if abs(v) < angle_thres:
            continue

        part_name = body_parts_korean['_'.join(k.split('_')[:-1])]
        particle = '을' if check_if_end_consonant(part_name) else '를'
        command = '펴주세요.' if v > 0 else '굽혀주세요.'
        feedback.append(f'{part_name}{particle} 좀 더 {command}')

    # distance feedback 생성
    maximum_save = {
        'right_max_key': '',
        'right_max_value': distance_thres,
        'left_max_key': '',
        'left_max_value': distance_thres
    }

    for k, v in diffs['distance_difference'].items():
        if abs(v) < distance_thres:
            continue

        part1, part2, _ = k.split()
        if part1.split('_')[1] == part2.split('_')[1]:
            common = body_parts_korean[part1.split('_')[1]]
            particle = '을' if check_if_end_consonant(common) else '를'
            command = '떼주세요.' if v > 0 else '붙이세요.'
            feedback.append(f'두 {common}{particle} 좀 더 {command}')
        else:
            # part_name_1 = body_parts_korean[part1]
            # part_name_2 = body_parts_korean[part2]
            # if v > 0:
            #     command = '떨어트리세요.'
            #     particle_1 = '과' if check_if_end_consonant(part_name_1) else '와'
            #     particle_2 = '을' if check_if_end_consonant(part_name_2) else '를'
            #     feedback.append(f'{part_name_1}{particle_1} {part_name_2}{particle_2} 좀 더 {command}')
            # else:
            #     command = '붙이세요.'
            #     particle_1 = '을' if check_if_end_consonant(part_name_1) else '를'
            #     feedback.append(f'{part_name_1}{particle_1} {part_name_2}쪽으로 좀 더 {command}')

            if part1.startswith('left') and abs(v) > abs(maximum_save['left_max_value']):
                maximum_save['left_max_key'] = k
                maximum_save['left_max_value'] = v
            
            if part1.startswith('right') and abs(v) > abs(maximum_save['right_max_value']):
                maximum_save['right_max_key'] = k
                maximum_save['right_max_value'] = v
    
    if maximum_save['right_max_key']:
        part1, part2, _ = maximum_save['right_max_key'].split()
        part_name_1 = body_parts_korean[part1]
        part_name_2 = body_parts_korean[part2]

        if maximum_save['right_max_value'] > 0:
            command = '떨어트리세요.'
            particle_1 = '과' if check_if_end_consonant(part_name_1) else '와'
            particle_2 = '을' if check_if_end_consonant(part_name_2) else '를'
            feedback.append(f'{part_name_1}{particle_1} {part_name_2}{particle_2} 좀 더 {command}')
        else:
            command = '붙이세요.'
            particle_1 = '을' if check_if_end_consonant(part_name_1) else '를'
            feedback.append(f'{part_name_1}{particle_1} {part_name_2}쪽으로 좀 더 {command}')
    
    if maximum_save['left_max_key']:
        part1, part2, _ = maximum_save['left_max_key'].split()
        part_name_1 = body_parts_korean[part1]
        part_name_2 = body_parts_korean[part2]

        if maximum_save['left_max_value'] > 0:
            command = '떨어트리세요.'
            particle_1 = '과' if check_if_end_consonant(part_name_1) else '와'
            particle_2 = '을' if check_if_end_consonant(part_name_2) else '를'
            feedback.append(f'{part_name_1}{particle_1} {part_name_2}{particle_2} 좀 더 {command}')
        else:
            command = '붙이세요.'
            particle_1 = '을' if check_if_end_consonant(part_name_1) else '를'
            feedback.append(f'{part_name_1}{particle_1} {part_name_2}쪽으로 좀 더 {command}')
    

    # 상대 위치 피드백 생성
    target_features = []
    for k, v in diffs['relative_pos_difference'].items():
        part1, part2, _ = k.split()
        part1 = body_parts_korean[part1]
        part2 = body_parts_korean[part2]

        if 'shoulder' in k:
            y_diff = v['y']['diff']
            z_diff = v['z']['diff']

            if abs(y_diff) > rel_thres:
                if y_diff > 0:
                    feedback.append("오른쪽 어깨를 더 내려주세요.")
                else:
                    feedback.append("왼쪽 어깨를 더 내려주세요.")
            
            if abs(z_diff) > rel_thres:
                if z_diff < 0:
                    feedback.append("왼쪽 어깨가 더 앞으로 나와야합니다.")
                else:
                    feedback.append("오른쪽 어깨가 더 앞으로 나와야합니다.")
        else:
            z_diff = v['z']['diff']
            if z_diff <= rel_thres: continue

            if z_diff > 0:
                feedback.append(f"{part1}이 좀 더 몸의 앞쪽에 있어야합니다.")
            else:
                feedback.append(f"{part1}이 좀 더 몸의 뒤쪽에 있어야합니다.")

    return feedback




if __name__ == "__main__":
    img_path_1 = './images/jun_ude.jpg'
    img_path_2 = './images/sy_shell.jpg'


    import sys
    sys.path.append("./")
    from dance_scoring import detector, scoring
    import config
    from feedback.pose_compare import extract_pose_world_landmarks


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
    result_1 = extract_pose_world_landmarks(landmark_1)
    result_2 = extract_pose_world_landmarks(landmark_2)
    differences = json_to_prompt_3(result_1, result_2)
    feedback_lst = get_korean_feedback_posescript(
        differences, angle_thres=25, distance_thres=0.15
    )
    for feedback in feedback_lst:
        print(feedback)


# if __name__ == "__main__":
#     import sys
#     sys.path.append("./")
#     from dance_scoring import detector
#     from feedback.pose_compare import extract_pose_world_landmarks

#     det = detector.PoseDetector()

#     img_path = './images/jun_ude.jpg'
#     landmark, _, _, _ = det.get_image_landmarks(img_path)
#     result = extract_pose_world_landmarks(landmark)

#     # pose feedback
#     feedback_module = FramePose3D(result)
#     print("################ HEAD ################")
#     print("고개가 숙여진/펴진 정도: ", feedback_module.get_head_angle_1())
#     print("고개가 오른/왼쪽으로 얼마나 숙여졌는지: ", feedback_module.get_head_angle_2())
#     print("시선이 오른/왼쪽으로 얼마나 돌아갔는지: ", feedback_module.get_eye_direction())
    
#     print("################ BODY ################")
#     print("허리가 앞으로 숙여진 정도: ", feedback_module.get_waist_angle_1())
#     print("사용자 몸이 돌아간 정도: ", feedback_module.get_waist_angle_2())

#     print("################ LEFT ARM ################")
#     print("왼팔 굽혀진 정도 : ", feedback_module.get_left_elbow_angle())
#     print("왼팔 높이: ", feedback_module.get_left_arm_height())
#     print("왼팔 방향: ", feedback_module.get_left_arm_dir())

#     print("################ RIGHT ARM ################")
#     print("오른팔 굽혀진 정도 : ", feedback_module.get_right_elbow_angle())
#     print("오른팔 높이: ", feedback_module.get_right_arm_height())
#     print("오른팔 방향: ", feedback_module.get_right_arm_dir())

#     print("################ LEFT LEG ################")
#     print("왼다리 굽혀진 정도 : ", feedback_module.get_left_knee_angle())
#     print("왼다리 높이: ", feedback_module.get_left_leg_height())
#     print("왼다리 방향: ", feedback_module.get_left_leg_dir())

#     print("################ RIGHT LEG ################")
#     print("오른다리 굽혀진 정도 : ", feedback_module.get_right_knee_angle())
#     print("오른다리 높이: ", feedback_module.get_right_leg_height())
#     print("오른다리 방향: ", feedback_module.get_right_leg_dir())
#     print("다리가 벌어진 정도: ", feedback_module.get_leg_angle())