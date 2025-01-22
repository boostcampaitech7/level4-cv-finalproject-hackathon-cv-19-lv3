import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean, cosine
import random
import os

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints_with_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose 추론 수행
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            keypoints = []
            for landmark in result.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints_list.append(np.array(keypoints).flatten())
            frames.append(frame)  # 원본 프레임 저장

    cap.release()
    return np.array(keypoints_list), frames

def l2_normalize(keypoints):
    norms = np.linalg.norm(keypoints, axis=1, keepdims=True)
    normalized_keypoints = keypoints / (norms + 1e-10)
    return normalized_keypoints

def filter_keypoints(keypoints, indices):
    """ 특정 keypoint 인덱스만 필터링 """
    return keypoints[:, np.array(indices).flatten()]

def calculate_similarity_with_visualization(keypoints1, keypoints2):
    distance, pairs = fastdtw(keypoints1, keypoints2, dist=euclidean)

    cosine_similarities = []
    euclidean_distances = []

    for idx1, idx2 in pairs:
        cosine_similarity = 1 - cosine(keypoints1[idx1], keypoints2[idx2])
        euclidean_distance = euclidean(keypoints1[idx1], keypoints2[idx2])

        cosine_similarities.append(cosine_similarity)
        euclidean_distances.append(euclidean_distance)

    average_cosine_similarity = np.mean(cosine_similarities)
    average_euclidean_distance = np.mean(euclidean_distances)

    return distance, average_cosine_similarity, average_euclidean_distance, pairs

def calculate_oks(keypoints1, keypoints2, pairs, sigma=0.1):
    oks_values = []
    for idx1, idx2 in pairs:
        kp1 = keypoints1[idx1].reshape(-1, 3)
        kp2 = keypoints2[idx2].reshape(-1, 3)
        
        # Euclidean distance for each keypoint
        d = np.linalg.norm(kp1[:, :2] - kp2[:, :2], axis=1)
        
        # Object Keypoint Similarity (OKS)
        oks = np.exp(-d**2 / (2 * (sigma**2)))
        oks_values.append(np.mean(oks))

    average_oks = np.mean(oks_values)
    return average_oks

def save_random_pair_frames(pairs, frames1, frames2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    random_pair = random.choice(pairs)
    idx1, idx2 = random_pair

    if idx1 < len(frames1) and idx2 < len(frames2):
        frame1 = frames1[idx1]
        frame2 = frames2[idx2]

        frame1_path = os.path.join(output_dir, f"frame_video1_{idx1}.jpg")
        frame2_path = os.path.join(output_dir, f"frame_video2_{idx2}.jpg")

        cv2.imwrite(frame1_path, frame1)
        cv2.imwrite(frame2_path, frame2)

        print(f"Saved frames: {frame1_path}, {frame2_path}")

# 비디오 경로
video1 = "video1.mp4"
video2 = "challenge.mp4"

# keypoints 및 프레임 추출
keypoints1, frames1 = extract_keypoints_with_frames(video1)
keypoints2, frames2 = extract_keypoints_with_frames(video2)

# 특정 keypoints 인덱스 (예: 어깨와 엉덩이만 사용)
selected_indices = [11, 12, 23, 24]  # 어깨와 엉덩이 keypoints
keypoints1_filtered = filter_keypoints(keypoints1, selected_indices)
keypoints2_filtered = filter_keypoints(keypoints2, selected_indices)

# keypoints L2 정규화
keypoints1_filtered = l2_normalize(keypoints1_filtered)
keypoints2_filtered = l2_normalize(keypoints2_filtered)

# 유사도 및 시각화 데이터 계산
distance, avg_cosine, avg_euclidean, pairs = calculate_similarity_with_visualization(
    keypoints1_filtered, keypoints2_filtered
)

# OKS 계산
oks = calculate_oks(keypoints1_filtered, keypoints2_filtered, pairs)

# 랜덤한 매칭된 프레임 저장
output_dir = "output_frames"
save_random_pair_frames(pairs, frames1, frames2, output_dir)

print(f"{video1} with {video2}")
print(f"Initial distance: {distance}")
print(f"Average Cosine Similarity: {avg_cosine}")
print(f"Average Euclidean Distance: {avg_euclidean}")
print(f"Average OKS: {oks}")
