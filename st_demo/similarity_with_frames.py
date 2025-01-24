import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean, cosine
import os
from keypoint_map import SELECTED_SIGMAS, SELECTED_KEYPOINTS
from random import choice


# PCK 값 계산 함수
def pck(gt, preds, threshold=0.1, ignore_z=False):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    target_end = 3-ignore_z
    distance = np.linalg.norm(gt[:, :target_end] - preds[:, :target_end], axis=1)
    matched = distance < threshold
    pck_score = np.mean(matched)
    return pck_score, matched

# OKS 값 계산 함수
def oks(gt, preds, ignore_z=False):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    target_end = 3-ignore_z
    sigmas = np.array(SELECTED_SIGMAS)
    distance = np.linalg.norm(gt[:, :target_end] - preds[:, :target_end], axis=1)

    kp_c = sigmas * 2
    return np.mean(np.exp(-(distance ** 2) / (2 * (kp_c ** 2))))


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

def calculate_similarity_with_visualization(keypoints1, keypoints2):
    # FastDTW로 DTW 거리와 매칭된 인덱스 쌍(pairs) 계산
    distance, pairs = fastdtw(keypoints1, keypoints2, dist=normalize_landmarks_to_range)

    cosine_similarities = []
    euclidean_distances = []
    oks_list = []
    pck_list = []

    for idx1, idx2 in pairs:
        # 각 프레임에서 keypoints의 Cosine Similarity와 Euclidean Distance 계산
        kp1 = keypoints1[idx1]  # shape: (keypoint_count, 3)
        kp2 = keypoints2[idx2]  # shape: (keypoint_count, 3)

        # 벡터 차원을 유지한 상태로 유사도 계산
        frame_cosine_similarities = []
        frame_euclidean_distances = []
        oks_list.append(oks(kp1, kp2))
        pck_list.append(pck(kp1, kp2, threshold=0.1)[0])

        for p1, p2 in zip(kp1, kp2):
            # 코사인 유사도 계산
            cosine_similarity = 1 - cosine(p1, p2)
            frame_cosine_similarities.append(cosine_similarity)

            # 유클리드 거리 계산
            euclidean_distance = euclidean(p1, p2)
            frame_euclidean_distances.append(euclidean_distance)

        # 각 프레임의 평균 값을 저장
        cosine_similarities.append(np.mean(frame_cosine_similarities))
        euclidean_distances.append(np.mean(frame_euclidean_distances))

    # 전체 프레임에 대한 평균 값을 계산
    average_cosine_similarity = np.mean(cosine_similarities)
    average_euclidean_distance = np.mean(euclidean_distances)
    average_oks = np.mean(oks_list)
    average_pck = np.mean(pck_list)

    return distance, average_cosine_similarity, average_euclidean_distance, average_oks, average_pck, pairs

def get_center_pair_frames(pairs, keypoint2_index):
    # keypoint2_index와 매칭된 keypoint1의 모든 인덱스를 찾음
    matching_pairs = [pair for pair in pairs if pair[1] == keypoint2_index]
    num_total_pairs = len(matching_pairs)

    if not matching_pairs:
        print(f"No matching pairs found for keypoint2 index: {keypoint2_index}")
        return (0, 0)

    # keypoint1 매칭된 프레임 중 랜덤 번호 하나 return
    # return choice([idx1 for idx1, idx2 in matching_pairs])

    # 매칭된 프레임 중 시간축 기준 중간에 위치하는 것을 추출
    return matching_pairs[num_total_pairs // 2][0]

def get_all_pair_frames(pairs, keypoint2_index):
    # keypoint2_index와 매칭된 keypoint1의 모든 인덱스를 찾음
    matching_pairs = [pair for pair in pairs if pair[1] == keypoint2_index]

    if not matching_pairs:
        print(f"No matching pairs found for keypoint2 index: {keypoint2_index}")
        return (0, 0)

    # keypoint1 매칭된 프레임 번호 모두 return
    return matching_pairs


def make_cosine_similarity(avg_cosine):
    # cos = (avg_cosine * 1000) % 100
    cos = (1 + avg_cosine) / 2 * 100
    return cos

def make_euclidean_similarity(avg_euclidean):
    euc = 100 - (avg_euclidean * 100)
    return euc