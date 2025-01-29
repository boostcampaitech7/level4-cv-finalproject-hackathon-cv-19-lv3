import math
import json
import numpy as np
import random

def generate_feedback(feature_differences, threshold = 30):
    """
    Generate feedback based on the feature differences provided.

    Parameters:
    - feature_differences (dict): Dictionary of feature differences.

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
                feedback_dict[feature.replace("_angle_difference", "")] = f"Lower your {side} arm." if difference > 0 else f"Raise your {side} arm."
            elif feature in ["left_elbow_angle_difference", "right_elbow_angle_difference"]:
                side = "left" if "left" in feature else "right"
                feedback_dict[feature.replace("_angle_difference", "")] = f"Straighten your {side} elbow." if difference > 0 else f"Bend your {side} elbow."
            elif feature in ["left_leg_angle_difference", "right_leg_angle_difference"]:
                side = "left" if "left" in feature else "right"
                feedback_dict[feature.replace("_angle_difference", "")] = f"Lower your {side} leg." if difference > 0 else f"Raise your {side} leg."
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
            if feature == "head_difference":
                feedback_dict["head"] = "머리를 왼쪽으로 기울이세요." if difference > 0 else "머리를 오른쪽으로 기울이세요."
            elif feature == "shoulder_difference":
                if difference > 0:
                    random_choice = random.choice(["왼쪽 어깨를 내리세요.", "오른쪽 어깨를 올리세요."])
                else:
                    random_choice = random.choice(["왼쪽 어깨를 올리세요.", "오른쪽 어깨를 내리세요."])
                feedback_dict["shoulder"] = random_choice
            elif feature in ["left_arm_angle_difference", "right_arm_angle_difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 팔을 내리세요." if difference > 0 else f"{side} 팔을 올리세요."
            elif feature in ["left_elbow_angle_difference", "right_elbow_angle_difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 팔꿈치를 펴세요." if difference > 0 else f"{side} 팔꿈치를 구부리세요."
            elif feature in ["left_leg_angle_difference", "right_leg_angle_difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 다리를 내리세요." if difference > 0 else f"{side} 다리를 올리세요."
            elif feature in ["left_knee_angle_difference", "right_knee_angle_difference"]:
                side = "왼쪽" if "left" in feature else "오른쪽"
                feedback_dict[feature.replace("_angle_difference", "")] = f"{side} 무릎을 펴세요." if difference > 0 else f"{side} 무릎을 구부리세요."
    
    # 기준값을 초과한 특징이 없으면 기본 성공 메시지 반환
    if not feedback_dict:
        return {"perfect_msg": "훌륭합니다! 자세가 완벽합니다!"}
    
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


def json_to_prompt(target_landmarks_json_path, compare_landmarks_json_path, result_folder="./prompts", threshold=30):
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
        "left_arm_angle_difference": int(pose1.get_left_arm_angle() - pose2.get_left_arm_angle()),
        "right_arm_angle_difference": -int(pose1.get_right_arm_angle() - pose2.get_right_arm_angle()),
        "left_elbow_angle_difference": int(pose1.get_left_elbow_angle() - pose2.get_left_elbow_angle()),
        "right_elbow_angle_difference": int(pose1.get_right_elbow_angle() - pose2.get_right_elbow_angle()),
        "left_leg_angle_difference": int(pose1.get_left_leg_angle() - pose2.get_left_leg_angle()),
        "right_leg_angle_difference": int(pose1.get_right_leg_angle() - pose2.get_right_leg_angle()),
        "left_knee_angle_difference": int(pose1.get_left_knee_angle() - pose2.get_left_knee_angle()),
        "right_knee_angle_difference": -int(pose1.get_right_knee_angle() - pose2.get_right_knee_angle()),
    }
    # natural_language_json = generate_feedback(result_json, threshold=threshold)
    natural_language_json = generate_korean_feedback(result_json, threshold=threshold)
    return result_json, natural_language_json

    # JSON 파일명 생성
    # if not os.path.exists(result_folder):
    #     os.mkdir(result_folder)
    # target_data_name = list(Path(target_landmarks_json_path).parts)[-2]
    # compare_data_name = list(Path(compare_landmarks_json_path).parts)[-2]
    # json_file_name = f"{target_data_name.split('_')[0]}_{compare_data_name.split('_')[0]}_{target_data_name.split('_')[-1]}.json"
    # json_file_path = os.path.join(result_folder, json_file_name)

    # # JSON 파일 저장
    # with open(json_file_path, 'w') as f:
    #     json.dump(result_json, f, indent=4)