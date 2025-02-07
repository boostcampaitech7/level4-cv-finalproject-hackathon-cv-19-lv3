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

            if feature == "head_difference":
                feedback_dict["head"] = f"머리를 {modifier} 왼쪽으로 기울여주세요." if difference > 0 else f"머리를 {modifier} 오른쪽으로 기울여주세요."

            elif feature == "shoulder_difference":
                modifier = "너무" if abs(difference) > threshold*2 else "약간"
                if difference > 0:
                    feedback_dict["shoulder"] = f"왼쪽 어깨를 조금 내려주세요."
                else:
                    feedback_dict["shoulder"] = f"오른쪽 어깨를 조금 내려주세요."

            elif feature in ["left_arm_angle_difference", "right_arm_angle_difference"]:
                modifier = "전혀" if abs(difference) > threshold*2 else "약간"
                side = "왼쪽" if "left" in feature else "오른쪽"
                to = "오른쪽" if difference > 0 else "왼쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 팔을 좀 더 {to}으로 움직여주세요."
            
            elif feature in ["left_elbow_angle_difference", "right_elbow_angle_difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 팔꿈치를 {modifier} 펴주세요." if difference > 0 else f"{side} 팔꿈치를 {modifier} 구부려주세요."

            elif feature in ["left_leg_angle_difference", "right_leg_angle_difference"]:
                modifier = "전혀" if abs(difference) > threshold*2 else "약간"
                side = "왼쪽" if "left" in feature else "오른쪽"
                to = "오른쪽" if difference > 0 else "왼쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 다리를 좀 더 {to}으로 움직여주세요."
                    
            elif feature in ["left_knee_angle_difference", "right_knee_angle_difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 무릎을 {modifier} 펴주세요." if difference > 0 else f"{side} 무릎을 {modifier} 구부려주세요."
    
    # 기준값을 초과한 특징이 없으면 기본 성공 메시지 반환
    if not feedback_dict:
        perfect_msg = [
            "동작 하나하나가 정말 정확하고 힘이 느껴져요!",
            "자연스럽고 부드러운 흐름이 너무 멋져요!",
            "디테일이 살아 있어요!",
            "춤선이 정말 아름답고 완벽하게 살아 있네요!",
            "스텝이 너무 정확하고 깔끔해서 감탄했어요!"
        ]
        return {"perfect_msg": random.choice(perfect_msg)}
    
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
        return calculate_two_points_angle(self.left_shoulder, self.left_elbow)

    def get_right_arm_angle(self):
        return calculate_two_points_angle(self.right_shoulder, self.right_elbow)

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
        "head_difference": int(pose1.get_ear_height_difference() - pose2.get_ear_height_difference()),
        "shoulder_difference": int(pose1.get_shoulder_height_difference() - pose2.get_shoulder_height_difference()),
        "left_arm_angle_difference": int(abs(pose1.get_left_arm_angle()) - abs(pose2.get_left_arm_angle())),
        "right_arm_angle_difference": int(abs(pose1.get_right_arm_angle()) - abs(pose2.get_right_arm_angle())),
        "left_elbow_angle_difference": int(pose1.get_left_elbow_angle() - pose2.get_left_elbow_angle()),
        "right_elbow_angle_difference": int(pose1.get_right_elbow_angle() - pose2.get_right_elbow_angle()),
        "left_leg_angle_difference": int(abs(pose1.get_left_leg_angle()) - abs(pose2.get_left_leg_angle())),
        "right_leg_angle_difference": int(abs(pose1.get_right_leg_angle()) - abs(pose2.get_right_leg_angle())),
        "left_knee_angle_difference": int(pose1.get_left_knee_angle() - pose2.get_left_knee_angle()),
        "right_knee_angle_difference": int(pose1.get_right_knee_angle() - pose2.get_right_knee_angle()),
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


if __name__ == "__main__":
    # import sys
    # sys.path.append("./")
    # from dance_scoring import detector
    # from feedback.pose_compare import extract_pose_world_landmarks

    # det = detector.PoseDetector()

    # img_path = './images/jun_ude.jpg'
    # landmark, _, _, _ = det.get_image_landmarks(img_path)
    # result = extract_pose_world_landmarks(landmark)

    # # pose feedback
    # feedback_module = FramePose3D(result)
    # print("################ HEAD ################")
    # print("고개가 숙여진/펴진 정도: ", feedback_module.get_head_angle_1())
    # print("고개가 오른/왼쪽으로 얼마나 숙여졌는지: ", feedback_module.get_head_angle_2())
    # print("시선이 오른/왼쪽으로 얼마나 돌아갔는지: ", feedback_module.get_eye_direction())
    
    # print("################ BODY ################")
    # print("허리가 앞으로 숙여진 정도: ", feedback_module.get_waist_angle_1())
    # print("사용자 몸이 돌아간 정도: ", feedback_module.get_waist_angle_2())

    # print("################ LEFT ARM ################")
    # print("왼팔 굽혀진 정도 : ", feedback_module.get_left_elbow_angle())
    # print("왼팔 높이: ", feedback_module.get_left_arm_height())
    # print("왼팔 방향: ", feedback_module.get_left_arm_dir())

    # print("################ RIGHT ARM ################")
    # print("오른팔 굽혀진 정도 : ", feedback_module.get_right_elbow_angle())
    # print("오른팔 높이: ", feedback_module.get_right_arm_height())
    # print("오른팔 방향: ", feedback_module.get_right_arm_dir())

    # print("################ LEFT LEG ################")
    # print("왼다리 굽혀진 정도 : ", feedback_module.get_left_knee_angle())
    # print("왼다리 높이: ", feedback_module.get_left_leg_height())
    # print("왼다리 방향: ", feedback_module.get_left_leg_dir())

    # print("################ RIGHT LEG ################")
    # print("오른다리 굽혀진 정도 : ", feedback_module.get_right_knee_angle())
    # print("오른다리 높이: ", feedback_module.get_right_leg_height())
    # print("오른다리 방향: ", feedback_module.get_right_leg_dir())
    # print("다리가 벌어진 정도: ", feedback_module.get_leg_angle())
    print(generate_korean_feedback({'head_difference': -2, 'shoulder_difference': 4, 'left_arm_angle_difference': 19, 'right_arm_angle_difference': 30, 'left_elbow_angle_difference': 35, 'right_elbow_angle_difference': -34, 'left_leg_angle_difference': 0, 'right_leg_angle_difference': -1, 'left_knee_angle_difference': -4, 'right_knee_angle_difference': -3}))