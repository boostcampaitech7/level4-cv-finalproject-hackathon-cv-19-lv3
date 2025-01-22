import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean, cosine

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
                # print([landmark.x, landmark.y, landmark.z])
            keypoints_list.append(np.array(keypoints).flatten())
        # else:
            # keypoints_list.append(np.zeros(99))
    
    cap.release()
    return np.array(keypoints_list)

def l2_normalize(keypoints):
    norms = np.linalg.norm(keypoints, axis=1, keepdims=True)
    normalized_keypoints = keypoints / (norms + 1e-10)
    return normalized_keypoints

def calculate_similarity(keypoints1, keypoints2):
    distance, pair = fastdtw(keypoints1, keypoints2, dist=euclidean)

    cosine_similarities = []
    euclidean_distances = []
    for idx1, idx2 in pair:
        cosine_similarity = 1 - cosine(keypoints1[idx1], keypoints2[idx2])  # 코사인 유사도 계산
        euclidean_distance = euclidean(keypoints1[idx1], keypoints2[idx2])  # 유클리드 거리 계산

        cosine_similarities.append(cosine_similarity)
        euclidean_distances.append(euclidean_distance)

        # Step 4: 평균 유사도 및 거리 계산
    average_cosine_similarity = np.mean(cosine_similarities)
    average_euclidean_distance = np.mean(euclidean_distances)

    return distance, average_cosine_similarity, average_euclidean_distance

# 비디오 경로
video1 = "video1.mp4"
video2 = "video2.mp4"

# keypoints 추출
keypoints1 = extract_keypoints(video1)
keypoints2 = extract_keypoints(video2)
# print(f'length of video is video1:{len(keypoints1)} and video2:{len(keypoints2)}')

# keypoints l2정규화
keypoints1 = l2_normalize(keypoints1)
keypoints2 = l2_normalize(keypoints2)

# 유사도 계산
distance, avg_cosine, avg_euclidean = calculate_similarity(keypoints1, keypoints2)

print(f"Initial distance: {distance}")
print(f"Average Cosine Similarity: {avg_cosine}")
print(f"Average Euclidean Distance: {avg_euclidean}")