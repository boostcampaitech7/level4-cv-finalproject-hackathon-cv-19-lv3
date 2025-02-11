import cv2
import json
import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean, cosine
import os

def l2_normalize(keypoints):
    norms = np.linalg.norm(keypoints, axis=2, keepdims=True)
    normalized_keypoints = keypoints / (norms + 1e-10)
    return normalized_keypoints

def filter_keypoints(keypoints, indices):
    return keypoints[:, np.array(indices), :]

# keypoint1의 프레임에 맞춰서 keypoint2의 프레임을 정규화
# 하나의 프레임만 진행해야함 -> 1의 프레임에 맞춰서 2의 프레임을 정규화 하므로
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
        keypoints1 = keypoints1[:, :3]  # (33, 4) → (33, 3)
        keypoints2 = (keypoints2[:, :3] - min2) / (max2 - min2 + eps) * (max1 - min1) + min1

        # Combine normalized coordinates with the original visibility values
        keypoints2 = np.hstack((keypoints2, keypoints2[:, 3:4]))

        return np.linalg.norm(keypoints1 - keypoints2)

def normalize_landmarks_to_pair(landmarks_np_1, landmarks_np_2, eps=1e-7):
    """
    Normalize landmarks2 to match the coordinate range of landmarks1.

    Parameters:
        landmarks_np_1 (numpy array): Keypoints array for the first pose (num_selected_point, 4).
        landmarks_np_2 (numpy array): Keypoints array for the second pose (num_selected_point, 4).

    Returns:
        numpy array: Normalized landmarks2 matching the range of landmarks1.
    """
    # Calculate min and max for landmarks1 and landmarks2
    min1 = np.min(landmarks_np_1[:, :3], axis=0)  # (x_min, y_min, z_min) for landmarks1
    max1 = np.max(landmarks_np_1[:, :3], axis=0)  # (x_max, y_max, z_max) for landmarks1

    min2 = np.min(landmarks_np_2[:, :3], axis=0)  # (x_min, y_min, z_min) for landmarks2
    max2 = np.max(landmarks_np_2[:, :3], axis=0)  # (x_max, y_max, z_max) for landmarks2

    # Normalize landmarks2 to the range of landmarks1
    normalized_landmarks2 = (landmarks_np_2[:, :3] - min2) / (max2 - min2 + eps) * (max1 - min1) + min1

    # Combine normalized coordinates with the original visibility values
    normalized_landmarks2 = np.hstack((normalized_landmarks2, landmarks_np_2[:, 3:4]))

    return normalized_landmarks2

def calculate_similarity_with_visualization(keypoints1, keypoints2):
    # FastDTW로 DTW 거리와 매칭된 인덱스 쌍(pairs) 계산
    distance, pairs = fastdtw(keypoints1, keypoints2, dist=normalize_landmarks_to_range)

    cosine_similarities = []
    euclidean_similarities = []

    for idx1, idx2 in pairs:
        # 각 프레임에서 keypoints의 Cosine Similarity와 Euclidean Distance 계산
        kp1 = keypoints1[idx1]  # shape: (keypoint_count, 3)
        kp2 = keypoints2[idx2]  # shape: (keypoint_count, 3)
        
        if len(kp1) == 0 or len(kp2) == 0:
            cosine_similarities.append(-1)
            euclidean_similarities.append(-1)
            continue
        
        kp2 = normalize_landmarks_to_pair(kp1, kp2) # kp2를 kp1의 범위에 맞게 정규화

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

# COCO dataset에서 제공하는 OKS 계산을 위한 keypoint standard deviation (σ values)
SELECTED_SIGMAS = [
    0.026, # nose
    # 0.025, # left_eye_inner
    # 0.025, # left_eye
    # 0.025, # left_eye_outer
    # 0.025, # right_eye_inner
    # 0.025, # right_eye
    # 0.025, # right_eye_outer
    0.035, # left_ear
    0.035, # right_ear
    # 0.026, # mouth_left
    # 0.026, # mouth_right
    0.079, # left_shoulder
    0.079, # right_shoulder
    0.072, # left_elbow
    0.072, # right_elbow
    0.062, # left_wrist
    0.062, # right_wrist
    # 0.072, # left_pinky
    # 0.072, # right_pinky
    # 0.072, # left_index
    # 0.072, # right_index
    # 0.072, # left_thumb
    # 0.072, # right_thumb
    0.107, # left_hip
    0.107, # right_hip
    0.087, # left_knee
    0.087, # right_knee
    0.089, # left_ankle
    0.089, # right_ankle
    0.089, # left_heel
    0.089, # right_heel
    0.072, # left_foot_index
    0.072  # right_foot_index
]

# OKS 값 계산 함수
def oks(gt, preds):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    sigmas = np.array(SELECTED_SIGMAS)
    gt = np.array(gt)
    preds = np.array(preds)
    distance = np.linalg.norm(gt - preds, axis=1)

    kp_c = sigmas * 2
    return np.mean(np.exp(-(distance ** 2) / (2 * (kp_c ** 2))))


# PCK 값 계산 함수
def pck(gt, preds, threshold=0.1):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    gt = np.array(gt)
    preds = np.array(preds)
    distance = np.linalg.norm(gt - preds, axis=1)
    # distance = np.linalg.norm(gt[:, :3] - preds[:, :3], axis=1)
    matched = distance < threshold
    pck_score = np.mean(matched)
    return pck_score, matched

def calculate_similarity_with_visualization2(keypoints1, keypoints2):
    # FastDTW로 DTW 거리와 매칭된 인덱스 쌍(pairs) 계산
    distance, pairs = fastdtw(keypoints1, keypoints2, dist=normalize_landmarks_to_range)

    cosine_similarities = []
    euclidean_similarities = []
    weighted_distances = []
    oks_scores = []
    pck_scores = []

    for idx1, idx2 in pairs:
        # 각 프레임에서 keypoints의 Cosine Similarity와 Euclidean Distance 계산
        kp1 = keypoints1[idx1]  # shape: (keypoint_count, 4) (x, y, z, confidence)
        kp2 = keypoints2[idx2]  # shape: (keypoint_count, 4)

        if len(kp1) == 0 or len(kp2) == 0:
            cosine_similarities.append(0)
            euclidean_similarities.append(0)
            weighted_distances.append(0)
            oks_scores.append(0)
            pck_scores.append(0)
            continue

        # kp2 = normalize_landmarks_to_pair(kp1, kp2)  # kp2를 kp1의 범위에 맞게 정규화

        # 벡터 차원을 유지한 상태로 유사도 계산
        frame_cosine_similarities = []
        frame_euclidean_similarities = []
        frame_weighted_distances = []

        # 각 keypoint의 confidence 값을 가져오기
        # confidences1 = kp1[:, 3]  # P(pose1(x_i, y_i))
        # confidences2 = kp2[:, 3]
        # confidences = confidences1 * confidences2
        # confidence_sum = np.sum(confidences) + 1e-10  # 0으로 나누는 것을 방지

        for p1, p2 in zip(kp1[:], kp2[:]):
            # 코사인 유사도 계산
            # print(f'conf:{conf}')
            cosine_similarity = min(1, 1 - cosine(p1, p2)) # 0~1 사이의 범위 조절
            # print(f'cos:{cosine_similarity}')
            frame_cosine_similarities.append(cosine_similarity)

            # 유클리드 유사도 계산
            euclidean_similarity = max(0, 1 - euclidean(p1, p2)) # 0~1 사이의 범위 조절
            # print(f'euc:{euclidean_similarity}')
            # print()
            frame_euclidean_similarities.append(euclidean_similarity)

            # Weighted Distance 계산
            weighted_similarity = euclidean_similarity
            frame_weighted_distances.append(weighted_similarity)

        # 각 프레임의 평균 값을 저장
        cosine_similarities.append(np.mean(frame_cosine_similarities))
        euclidean_similarities.append(np.mean(frame_euclidean_similarities))
        # weighted_distances.append(np.sum(frame_weighted_distances) / confidence_sum)
        weighted_distances.append(np.mean(frame_weighted_distances))  # Normalize by confidence sum

        oks_score = oks(kp1, kp2)
        oks_scores.append(oks_score)

        # PCK 계산 (각 프레임별)
        pck_score, _ = pck(kp1, kp2)
        pck_scores.append(pck_score)

    # 전체 프레임에 대한 평균 값을 계산
    average_cosine_similarity = np.mean(cosine_similarities)
    average_euclidean_similarity = np.mean(euclidean_similarities)
    average_weighted_distance = np.mean(weighted_distances)
    average_oks = np.mean(oks_scores)
    average_pck = np.mean(pck_scores)

    return distance, average_cosine_similarity, average_euclidean_similarity, average_weighted_distance, average_oks, average_pck, pairs


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
# video1 = "../video/video_tiktok1.mp4" 
# video2 = "../video/video_reverse2.mp4"

# keypoints 및 프레임 추출
# keypoints1, frames1 = extract_keypoints_with_frames(video1)
# print(keypoints1.shape)
# keypoints2, frames2 = extract_keypoints_with_frames(video2)
# print(keypoints2.shape)

def load_pose_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['all_frames_points'], data['fps']

keypoints1, fps1 = load_pose_data("backend/data/250205105914/user.json")
keypoints2, fps2 = load_pose_data("backend/data/250205105914/user.json")

# 특정 keypoints 인덱스
# 전체 인덱스를 사용하는게 유사도 차이를 더욱 보여줌
# selected_indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] # 19개
# keypoints1 = filter_keypoints(keypoints1, selected_indices)
# keypoints2 = filter_keypoints(keypoints2, selected_indices)

# 유사도 및 시각화 데이터 계산
distance, avg_cosine, avg_euclidean, avg_weighted, average_oks, average_pck, pairs = calculate_similarity_with_visualization2(
    keypoints1, keypoints2
)

# 랜덤한 매칭된 프레임 저장
# output_dir = "../output_frames"
# save_random_pair_frames(pairs, frames1, frames2, output_dir, 100)

# print(f"{video1} pairing with {video2} in in 1-stage experiment")
# print(f"Average Cosine Similarity score: {avg_cosine}")
# print(f"Average Euclidean Similarity score: {avg_euclidean}")
# print(f"Average Weighted Similarity score: {avg_weighted}")
# print(f"Average OKS score: {average_oks}")
# print(f"Average PCK score: {average_pck}")

# print()

# print(f"Cos(0.3) + Euc(0.3) + OKS(0.4): {avg_cosine*0.3 + avg_euclidean*0.3 + average_oks*0.4}")
# print(f"Cos(0.4) + Euc(0.4) + OKS(0.2): {avg_cosine*0.4 + avg_euclidean*0.4 + average_oks*0.2}")
# print(f"Cos(0.5) + Euc(0.3) + OKS(0.2): {avg_cosine*0.5 + avg_euclidean*0.3 + average_oks*0.2}")
# print(f"Cos(0.8) + Euc(0.1) + OKS(0.1): {avg_cosine*0.8 + avg_euclidean*0.1 + average_oks*0.1}")

# print(f"Cos(0.3) + Euc(0.3) + PKS(0.4): {avg_cosine*0.3 + avg_euclidean*0.3 + average_pck*0.4}")
# print(f"Cos(0.4) + Euc(0.4) + PKS(0.2): {avg_cosine*0.4 + avg_euclidean*0.4 + average_pck*0.2}")
# print(f"Cos(0.5) + Euc(0.3) + PKS(0.2): {avg_cosine*0.5 + avg_euclidean*0.3 + average_pck*0.2}")
# print(f"Cos(0.8) + Euc(0.1) + PKS(0.1): {avg_cosine*0.8 + avg_euclidean*0.1 + average_pck*0.1}")

# print(f"Cos(0.3) + Euc(0.3) + OKS(0.2) + PCK(0.2): {avg_cosine*0.3 + avg_euclidean*0.3 + average_oks*0.2 + average_pck*0.2}")
# print(f"Cos(0.4) + Euc(0.2) + OKS(0.2) + PCK(0.2): {avg_cosine*0.4 + avg_euclidean*0.2 + average_oks*0.2 + average_pck*0.2}")
# print(f"Cos(0.4) + Euc(0.4) + OKS(0.1) + PCK(0.1): {avg_cosine*0.4 + avg_euclidean*0.4 + average_oks*0.1 + average_pck*0.1}")
# print(f"Cos(0.5) + Euc(0.3) + OKS(0.1) + PCK(0.1): {avg_cosine*0.5 + avg_euclidean*0.3 + average_oks*0.1 + average_pck*0.1}")
# print(f"Cos(0.6) + Euc(0.2) + OKS(0.1) + PCK(0.1): {avg_cosine*0.6 + avg_euclidean*0.2 + average_oks*0.1 + average_pck*0.1}")
# print(f"Cos(0.65) + Euc(0.15) + OKS(0.1) + PCK(0.1): {avg_cosine*0.65 + avg_euclidean*0.15 + average_oks*0.1 + average_pck*0.1}")
# print(f"Cos(0.7) + Euc(0.1) + OKS(0.1) + PCK(0.1): {avg_cosine*0.7 + avg_euclidean*0.1 + average_oks*0.1 + average_pck*0.1}")
# print(f"Cos(0.8) + Euc(0.1) + OKS(0.05) + PCK(0.05): {avg_cosine*0.8 + avg_euclidean*0.1 + average_oks*0.05 + average_pck*0.05}")

if avg_cosine > 0.9 and avg_euclidean > 0.9:
    final_score = max(avg_cosine, avg_euclidean)
elif avg_cosine > 0.9 and avg_euclidean > 0.8:
    final_score = (avg_cosine+avg_euclidean) / 2
elif avg_cosine > 0.8 and avg_euclidean > 0.8:
    final_score = avg_cosine*0.7 + avg_euclidean*0.1 + average_oks*0.1 + average_pck*0.1
elif average_oks < 0.2 and average_pck < 0.1:
    final_score = min(average_oks, average_pck)
else:
    final_score = avg_cosine*0.3 + avg_euclidean*0.3 + average_oks*0.2 + average_pck*0.2

print(f"Final score is {final_score}")
