import cv2
import math
import json
import numpy as np
import matplotlib.pyplot as plt

def calculate_two_points_angle(point1, point2):
    vector = point2 - point1
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    angle = angle if angle >= 0 else angle + 360

    return angle

def calculate_three_points_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2 
    
    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    

    cos_angle = dot_product / (magnitude1 * magnitude2)

    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

class frame_pose:
    def __init__(self, landmarks_data):
        self.left_ear = np.array([
            landmarks_data['face']['7']['x'],
            landmarks_data['face']['7']['y']
        ])
        self.right_ear = np.array([
            landmarks_data['face']['8']['x'],
            landmarks_data['face']['8']['y']
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

if __name__ == "__main__":
    standard_landmarks = "right_1.json"
    compare_landmarks = "target_1.json"

    with open(standard_landmarks, 'r') as f:
        data1 = json.load(f)
    with open(compare_landmarks, 'r') as f:
        data2 = json.load(f)
    
    pose1 = frame_pose(data1)
    pose2 = frame_pose(data2)

    print("face difference :", int(pose1.get_ear_height_difference() - pose2.get_ear_height_difference()))
    print("shoudler difference :", int(pose1.get_shoulder_height_difference() - pose2.get_shoulder_height_difference()))
    print("left arm angle difference :", int(pose1.get_left_arm_angle() - pose2.get_left_arm_angle()))
    print("right arm angle difference :", -int(pose1.get_right_arm_angle() - pose2.get_right_arm_angle()))
    print("left elbow angle difference :", int(pose1.get_left_elbow_angle() - pose2.get_left_elbow_angle()))
    print("right elbow angle difference :", int(pose1.get_right_elbow_angle() - pose2.get_right_elbow_angle()))
    print("left leg angle difference :", int(pose1.get_left_leg_angle() - pose2.get_left_leg_angle()))
    print("right leg angle difference :", int(pose1.get_right_leg_angle() - pose2.get_right_leg_angle()))
    print("left knee angle difference :", int(pose1.get_left_knee_angle() - pose2.get_left_knee_angle()))
    print("right knee angle difference :", int(pose1.get_right_knee_angle() - pose2.get_right_knee_angle()))