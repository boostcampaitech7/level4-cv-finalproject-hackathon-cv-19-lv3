import os
import math
import json
import numpy as np
from fastapi.responses import JSONResponse

ORIGIN_JSON = "video.json"
USER_JSON = "user.json"

def calculate_two_points_angle(point1, point2):
    vector = point2 - point1
    angle = math.degrees(math.atan2(vector[1], vector[0]))

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
    def __init__(self, width, height, points):
        if points:
            self.left_ear = np.array([
                int(points[1][0] * width),
                int(points[1][1] * height)
            ])
            self.right_ear = np.array([
                int(points[2][0] * width),
                int(points[2][1] * height)
            ])
            self.left_shoulder = np.array([
                int(points[3][0] * width),
                int(points[3][1] * height)
            ])
            self.right_shoulder = np.array([
                int(points[4][0] * width),
                int(points[4][1] * height)
            ])
            self.left_elbow = np.array([
                int(points[5][0] * width),
                int(points[5][1] * height)
            ])
            self.right_elbow = np.array([
                int(points[6][0] * width),
                int(points[6][1] * height)
            ])
            self.left_wrist = np.array([
                int(points[7][0] * width),
                int(points[7][1] * height)
            ])
            self.right_wrist = np.array([
                int(points[8][0] * width),
                int(points[8][1] * height)
            ])
            self.left_hip = np.array([
                int(points[9][0] * width),
                int(points[9][1] * height)
            ])
            self.right_hip = np.array([
                int(points[10][0] * width),
                int(points[10][1] * height)
            ])
            self.left_knee = np.array([
                int(points[11][0] * width),
                int(points[11][1] * height)
            ])
            self.right_knee = np.array([
                int(points[12][0] * width),
                int(points[12][1] * height)
            ])
            self.left_ankle = np.array([
                int(points[13][0] * width),
                int(points[13][1] * height)
            ])
            self.right_ankle = np.array([
                int(points[14][0] * width),
                int(points[14][1] * height)
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
        return calculate_two_points_angle(self.left_hip, self.left_knee)

    def get_right_leg_angle(self):
        return calculate_two_points_angle(self.right_hip, self.right_knee)

    def get_left_knee_angle(self):
        return calculate_three_points_angle(self.left_hip, self.left_knee, self.left_ankle)

    def get_right_knee_angle(self):
        return calculate_three_points_angle(self.right_hip, self.right_knee, self.right_ankle)

def read_pose(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    all_frame_points = data.get("all_frames_points", [])
    width = data.get("width", 0)
    height = data.get("height", 0)

    return width, height, all_frame_points

async def get_frame_feedback_service(request):
    root_path = os.path.join("data", request.folder_id)
    target_path = os.path.join(root_path, ORIGIN_JSON)
    user_path = os.path.join(root_path, USER_JSON)

    width1, height1, all_frame_points1 = read_pose(target_path)
    width2, height2, all_frame_points2 = read_pose(user_path)

    # DTW로 frame에 맞는 target frame 계산
    user_frame = int(request.frame)
    target_frame = 0

    points1 = all_frame_points1[target_frame]
    points2 = all_frame_points2[user_frame]

    pose1 = frame_pose(width1, height1, points1)
    pose2 = frame_pose(width2, height2, points2)

    # 프롬프팅 후 Clova API 호출해서 feedback 받기

    return JSONResponse(content={"feedback": ""}, status_code=200)
