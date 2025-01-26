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
    real_keypoints_list = []
    normalize_keypoints_list = []
    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2] # 프레임 해상도 (정규화된 좌표를 픽셀 단위로 변환하기 위해서)

        # Pose 추론 수행
        result = pose.process(frame_rgb)

        real_landmarks = result.pose_world_landmarks # 3D 세계 좌표
        normalize_landmarks = result.pose_landmarks # 2D 정규화 좌표

        if real_landmarks and normalize_landmarks:
            real_keypoints = []
            normalize_keypoints = []
            for real_landmark, normalize_landmark in zip(real_landmarks.landmark, normalize_landmarks.landmark):
                real_x = real_landmark.x
                real_y = real_landmark.y
                real_z = real_landmark.z
                real_keypoints.append([real_x, real_y, real_z])

                normalize_x = normalize_landmark.x * width
                normalize_y = normalize_landmark.y * height
                normalize_z = normalize_landmark.z
                normalize_keypoints.append([normalize_x, normalize_y, normalize_z])

            real_keypoints_list.append(np.array(real_keypoints))
            normalize_keypoints_list.append(np.array(normalize_keypoints))
            frames.append(frame)  # 원본 프레임 저장

    cap.release()
    return np.array(real_keypoints_list), np.array(normalize_keypoints_list), frames

def l2_normalize(keypoints):
    norms = np.linalg.norm(keypoints, axis=2, keepdims=True)
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

def calculate_similarity_with_visualization(real_keypoints1, real_keypoints2, normalize_keypoints1, normalize_keypoints2):
    # FastDTW로 DTW 거리와 매칭된 인덱스 쌍(pairs) 계산
    distance, pairs = fastdtw(normalize_keypoints1, normalize_keypoints2, dist=normalize_landmarks_to_range)

    cosine_similarities = []
    euclidean_similarities = []

    for idx1, idx2 in pairs:
        # 각 프레임에서 keypoints의 Cosine Similarity와 Euclidean Distance 계산
        kp1 = real_keypoints1[idx1]  # shape: (keypoint_count, 3)
        kp2 = real_keypoints2[idx2]  # shape: (keypoint_count, 3)

        # 벡터 차원을 유지한 상태로 유사도 계산
        frame_cosine_similarities = []
        frame_euclidean_similarities = []

        for p1, p2 in zip(kp1, kp2):
            # 코사인 유사도 계산
            cosine_similarity = 1 - cosine(p1, p2)
            frame_cosine_similarities.append(cosine_similarity)

            # 유클리드 유사도 계산
            euclidean_similarity = 1- euclidean(p1, p2)
            frame_euclidean_similarities.append(euclidean_similarity)

        # 각 프레임의 평균 값을 저장
        cosine_similarities.append(np.mean(frame_cosine_similarities))
        euclidean_similarities.append(np.mean(frame_euclidean_similarities))

    # 전체 프레임에 대한 평균 값을 계산
    average_cosine_similarity = np.mean(cosine_similarities)
    average_euclidean_similarity = np.mean(euclidean_similarities)

    return distance, average_cosine_similarity, average_euclidean_similarity, pairs

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
video1 = "video_kdb1.mp4"
video2 = "video_kdb2.mp4"

# keypoints 및 프레임 추출
real_keypoints1, normalize_keypoints1, frames1 = extract_keypoints_with_frames(video1)
real_keypoints2, normalize_keypoints2, frames2 = extract_keypoints_with_frames(video2)

# 특정 keypoints 인덱스
# 전체 인덱스를 사용하는게 유사도 차이를 더욱 보여줌
# selected_indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] # 19개
# keypoints1 = filter_keypoints(keypoints1, selected_indices)
# keypoints2 = filter_keypoints(keypoints2, selected_indices)

# normalize_keypoints L2 정규화
normalize_keypoints1 = l2_normalize(normalize_keypoints1)
normalize_keypoints2 = l2_normalize(normalize_keypoints2)

# 유사도 및 시각화 데이터 계산
distance, avg_cosine, avg_euclidean, pairs = calculate_similarity_with_visualization(
    real_keypoints1, real_keypoints2, normalize_keypoints1, normalize_keypoints2
)

# 랜덤한 매칭된 프레임 저장
output_dir = "output_frames"
save_random_pair_frames(pairs, frames1, frames2, output_dir, 200)

print(f"{video1} pairing with {video2} in 2-stage")
print(f"Average Cosine Similarity score: {avg_cosine}")
print(f"Average Euclidean Similarity score: {avg_euclidean}")