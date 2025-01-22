import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        success, frame = cap.read() # 프레임 성공여부, 비디오 프레임
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose 추론 수행
        result = pose.process(frame)

        if result.pose_landmarks:
            keypoints = []
            for landmark in result.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints_list.append(np.array(keypoints).flatten())
        else:
            keypoints_list.append(np.zeros(99))
    
    cap.release()
    return np.array(keypoints_list)

def calculate_distance(keypoints1, keypoints2):
    distance, pair = fastdtw(keypoints1, keypoints2, dist=euclidean)
    return distance, pair

def calculate_normalized_similarity(keypoints1, keypoints2):
    distance, pair = fastdtw(keypoints1, keypoints2, dist=euclidean)
    max_distance_per_frame = np.sqrt(33 * 3)
    max_possible_distance = max_distance_per_frame * max(len(keypoints1), len(keypoints2))
    similarity = 1 - (distance / max_possible_distance)
    return similarity, pair

# def calculate_normalized_similarity(keypoints1, keypoints2)

video1 = "video1.mp4"
video2 = "video1.mp4"

# Keypoints 추출
keypoints1 = extract_keypoints(video1)
keypoints2 = extract_keypoints(video2)

# 유사도 계산
similarity_score, similarity_pair = calculate_normalized_similarity(keypoints1, keypoints2)

# 결과 출력
print(f"Normalized Similarity Score (0 to 1): {similarity_score}")

# Pose 객체 해제
pose.close()
