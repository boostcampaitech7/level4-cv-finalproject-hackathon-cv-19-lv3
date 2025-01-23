import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean, cosine
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
        height, width = frame.shape[:2] # 프레임 해상도 (정규화된 좌표를 픽셀 단위로 변환하기 위해서)

        # Pose 추론 수행
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            keypoints = []
            for landmark in result.pose_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z
                keypoints.append([x, y, z])
            # keypoints_list.append(np.array(keypoints).flatten())
            keypoints_list.append(np.array(keypoints))
            frames.append(frame)  # 원본 프레임 저장

    cap.release()
    return np.array(keypoints_list), frames

def l2_normalize(keypoints):
    norms = np.linalg.norm(keypoints, axis=1, keepdims=True)
    normalized_keypoints = keypoints / (norms + 1e-10)
    return normalized_keypoints

def filter_keypoints(keypoints, indices):
    return keypoints[:, np.array(indices), :]

def normalize_landmarks_to_range(keypoints1, keypoints2, eps=1e-7):
        """
        Normalize landmarks2 to match the coordinate range of landmarks1.

        Parameters:
            keypoints1 (numpy array): Keypoints array for the first pose (num_selected_point, 4).
            keypoints2 (numpy array): Keypoints array for the second pose (num_selected_point, 4).

        Returns:
            numpy array: Normalized landmarks2 matching the range of landmarks1.
        """
        # Calculate min and max for landmarks1 and landmarks2
        min1 = np.min(keypoints1[:, :3], axis=0)  # (x_min, y_min, z_min) for landmarks1
        max1 = np.max(keypoints1[:, :3], axis=0)  # (x_max, y_max, z_max) for landmarks1

        min2 = np.min(keypoints2[:, :3], axis=0)  # (x_min, y_min, z_min) for landmarks2
        max2 = np.max(keypoints2[:, :3], axis=0)  # (x_max, y_max, z_max) for landmarks2

        # Normalize landmarks2 to the range of landmarks1
        keypoints2 = (keypoints2[:, :3] - min2) / (max2 - min2 + eps) * (max1 - min1) + min1

        # Combine normalized coordinates with the original visibility values
        keypoints2 = np.hstack((keypoints2, keypoints2[:, 3:4]))

        return np.linalg.norm(keypoints1 - keypoints2)

def calculate_similarity_with_visualization(keypoints1, keypoints2):
    distance, pairs = fastdtw(keypoints1, keypoints2, dist=normalize_landmarks_to_range)
    
    # keypoint (k, 99)로 변환
    keypoints1 = keypoints1.reshape(keypoints1.shape[0], -1)
    keypoints2 = keypoints2.reshape(keypoints2.shape[0], -1)

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

def save_random_pair_frames(pairs, frames1, frames2, output_dir, keypoint2_index):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # keypoint2_index와 매칭된 keypoint1의 모든 인덱스를 찾음
    matching_pairs = [pair for pair in pairs if pair[1] == keypoint2_index]

    if not matching_pairs:
        print(f"No matching pairs found for keypoint2 index: {keypoint2_index}")
        return

    # keypoint2 인덱스에 해당하는 프레임 저장
    if keypoint2_index < len(frames2):
        frame2 = frames2[keypoint2_index]
        frame2_path = os.path.join(output_dir, f"frame_video2_{keypoint2_index}.jpg")
        cv2.imwrite(frame2_path, frame2)
        print(f"Saved keypoint2 frame: {frame2_path}")
    else:
        print(f"Invalid keypoint2 index: {keypoint2_index}")
        return

    # keypoint1 매칭된 모든 프레임 저장
    for idx1, idx2 in matching_pairs:
        if idx1 < len(frames1):
            frame1 = frames1[idx1]
            frame1_path = os.path.join(output_dir, f"frame_video1_{idx1}_matched_to_{keypoint2_index}.jpg")
            cv2.imwrite(frame1_path, frame1)
            print(f"Saved keypoint1 frame: {frame1_path}")
        else:
            print(f"Invalid keypoint1 index: {idx1}")


# 비디오 경로
video1 = "video1.mp4"
video2 = "video5.mp4"

# keypoints 및 프레임 추출
keypoints1, frames1 = extract_keypoints_with_frames(video1)
keypoints2, frames2 = extract_keypoints_with_frames(video2)
print(keypoints1.shape)
print(keypoints1[0])

# 특정 keypoints 인덱스
selected_indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] # 19개
keypoints1 = filter_keypoints(keypoints1, selected_indices)
keypoints2 = filter_keypoints(keypoints2, selected_indices)
print(keypoints1.shape)
print(keypoints1[0])

# keypoints L2 정규화
# keypoints1 = l2_normalize(keypoints1)
# keypoints2 = l2_normalize(keypoints2)

# 유사도 및 시각화 데이터 계산
distance, avg_cosine, avg_euclidean, pairs = calculate_similarity_with_visualization(
    keypoints1, keypoints2
)

# OKS 계산
# oks = calculate_oks(keypoints1, keypoints2, pairs)

# 랜덤한 매칭된 프레임 저장
output_dir = "output_frames"
# save_random_pair_frames(pairs, frames1, frames2, output_dir, 100)

print(f"{video1} with {video2}")
print(f"Initial distance: {distance}")
print(f"Average Cosine Similarity: {avg_cosine}")
print(f"Average Euclidean Distance: {avg_euclidean}")
# print(f"Average OKS: {oks}")