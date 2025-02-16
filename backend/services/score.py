import os
import h5py
import numpy as np
from fastdtw import fastdtw
from fastapi.responses import JSONResponse
from scipy.spatial.distance import euclidean, cosine
from config import logger
from constants import FilePaths, ResponseMessages, SELECTED_SIGMAS

def read_pose(h5_path: str):
    """Read pose landmarks from h5 file."""
    try:
        with h5py.File(h5_path, "r") as f:
            width = f["width"][()]
            height = f["height"][()]
            all_frame_points = f["all_frames_points"][()]

        return width, height, all_frame_points

    except Exception as e:
        raise ValueError(ResponseMessages.H5FILE_LOAD_FAIL.value.format(h5_path, str(e)))
    
def normalize_landmarks_to_range(keypoints1: np.ndarray, keypoints2: np.ndarray, eps: float = 1e-7) -> float:
    """Normalize pose landmarks origin video frame and user video frame."""
    min1 = np.min(keypoints1, axis=0)
    max1 = np.max(keypoints1, axis=0)
    min2 = np.min(keypoints2, axis=0)
    max2 = np.max(keypoints2, axis=0)

    keypoints1 = keypoints1
    keypoints2 = (keypoints2 - min2) / (max2 - min2 + eps) * (max1 - min1) + min1

    return np.linalg.norm(keypoints1 - keypoints2)

def normalize_landmarks_to_range_by_mean(all_landmarks_np_1, all_landmarks_np_2, eps=1e-7):
    """
    Normalize landmarks2 to match the coordinate range of landmarks1 using the average min and max values across frames.

    Parameters:
        all_landmarks_np_1 (numpy array): Keypoints array for the first pose (num_frames, num_selected_point, 4).
        all_landmarks_np_2 (numpy array): Keypoints array for the second pose (num_frames, num_selected_point, 4).

    Returns:
        numpy array: Normalized landmarks2 matching the range of landmarks1.
    """
    # Calculate the average min and max values for landmarks1
    min1_mean = np.mean(np.min(all_landmarks_np_1[:, :, :3], axis=1), axis=0)  # Average of per-frame (x_min, y_min, z_min)
    max1_mean = np.mean(np.max(all_landmarks_np_1[:, :, :3], axis=1), axis=0)  # Average of per-frame (x_max, y_max, z_max)

    # Calculate the average min and max values for landmarks2
    min2_mean = np.mean(np.min(all_landmarks_np_2[:, :, :3], axis=1), axis=0)  # Average of per-frame (x_min, y_min, z_min)
    max2_mean = np.mean(np.max(all_landmarks_np_2[:, :, :3], axis=1), axis=0)  # Average of per-frame (x_max, y_max, z_max)

    # Normalize all frames of landmarks2 to match the range of landmarks1
    normalized_landmarks2 = (all_landmarks_np_2[:, :, :3] - min2_mean) / (max2_mean - min2_mean + eps) * (max1_mean - min1_mean) + min1_mean

    # Combine normalized coordinates with the original visibility values
    normalized_landmarks2 = np.concatenate((normalized_landmarks2, all_landmarks_np_2[:, :, 3:4]), axis=2)

    return normalized_landmarks2

def oks(gt: np.ndarray, preds: np.ndarray) -> float:
    """Calculate Object Keypoint Similarity."""
    selected_sigma = np.array(SELECTED_SIGMAS)
    distance = np.linalg.norm(gt - preds, axis=1)
    kp_c = selected_sigma * 2

    return np.mean(np.exp(-(distance ** 2) / (2 * (kp_c ** 2))))

def pck(gt: np.ndarray, preds: np.ndarray, threshold: float = 0.1):
    """Calculate Percentage of Correct Keypoints."""
    distance = np.linalg.norm(gt - preds, axis=1)
    matched = distance < threshold

    return np.mean(matched), matched

def calculate_similarity(keypoints1: np.ndarray, keypoints2: np.ndarray):
    """Calculate similarity between two sequences of keypoints."""
    _, pairs = fastdtw(keypoints1, keypoints2, dist=normalize_landmarks_to_range)

    cosine_similarities = []
    euclidean_similarities = []
    weighted_distances = []
    oks_scores = []
    pck_scores = []

    for idx1, idx2 in pairs:
        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        if len(kp1) == 0 or len(kp2) == 0:
            cosine_similarities.append(0)
            euclidean_similarities.append(0)
            weighted_distances.append(0)
            oks_scores.append(0)
            pck_scores.append(0)
            continue

        frame_cosine_similarities = []
        frame_euclidean_similarities = []
        frame_weighted_distances = []

        for p1, p2 in zip(kp1, kp2):
            cosine_similarity = min(1, 1 - cosine(p1, p2))
            frame_cosine_similarities.append(cosine_similarity)

            euclidean_similarity = max(0, 1 - euclidean(p1, p2))
            frame_euclidean_similarities.append(euclidean_similarity)
            frame_weighted_distances.append(euclidean_similarity)

        cosine_similarities.append(np.mean(frame_cosine_similarities))
        euclidean_similarities.append(np.mean(frame_euclidean_similarities))
        weighted_distances.append(np.mean(frame_weighted_distances))

        oks_score = oks(kp1, kp2)
        oks_scores.append(oks_score)

        pck_score, _ = pck(kp1, kp2)
        pck_scores.append(pck_score)

    avg_cosine = np.mean(cosine_similarities)
    avg_euclidean = np.mean(euclidean_similarities)
    
    average_oks = np.mean(oks_scores)
    average_pck = np.mean(pck_scores)

    return avg_cosine, avg_euclidean, average_oks, average_pck

def calculate_score(keypoints1, keypoints2):
    """Calculate final score."""
    results = calculate_similarity(keypoints1, keypoints2)
    avg_cosine, avg_euclidean, average_oks, average_pck = results

    if avg_cosine > 0.9 and avg_euclidean > 0.9:
        final_score = max(avg_cosine, avg_euclidean)
    elif avg_cosine > 0.9 and avg_euclidean > 0.8:
        final_score = (avg_cosine + avg_euclidean) / 2
    elif avg_cosine > 0.8 and avg_euclidean > 0.8:
        final_score = avg_cosine * 0.7 + avg_euclidean * 0.1 + average_oks * 0.1 + average_pck * 0.1
    elif average_oks < 0.2 and average_pck < 0.1:
        final_score = min(average_oks, average_pck)
    else:
        final_score = avg_cosine * 0.3 + avg_euclidean * 0.3 + average_oks * 0.2 + average_pck * 0.2

    return int(final_score * 100)

async def get_score_service(folder_id: str):
    try:
        root_path = os.path.join("data", folder_id)
        target_path = os.path.join(root_path, FilePaths.ORIGIN_H5.value)
        user_path = os.path.join(root_path, FilePaths.USER_H5.value)

        # 원본 영상 및 유저 영상 포즈 정보 읽기
        _, _, all_frame_points1 = read_pose(target_path)
        _, _, all_frame_points2 = read_pose(user_path)

        # 원본 포즈와 유저 포즈 점수 계산
        score = calculate_score(all_frame_points1, all_frame_points2)

        logger.info(f"[{folder_id}] get score success: {score}")
        return JSONResponse(content={"score": score}, status_code=200)

    except Exception as e:
        logger.error(f"[{folder_id}] get score fail: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)