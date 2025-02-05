import os
import h5py
import numpy as np
from fastdtw import fastdtw
from fastapi.responses import JSONResponse
from scipy.spatial.distance import euclidean, cosine
from constants import FilePaths

def read_pose(h5_path):
    try:
        with h5py.File(h5_path, "r") as f:
            width = f["width"][()]
            height = f["height"][()]
            all_frame_points = f["all_frames_points"][()]

        return width, height, all_frame_points
    except Exception as e:
        raise ValueError(f"Failed to read pose data from {h5_path}: {str(e)}")
    
def normalize_landmarks_to_range(keypoints1: np.ndarray, keypoints2: np.ndarray, eps: float = 1e-7) -> float:
    min1 = np.min(keypoints1, axis=0)
    max1 = np.max(keypoints1, axis=0)
    min2 = np.min(keypoints2, axis=0)
    max2 = np.max(keypoints2, axis=0)

    keypoints1 = keypoints1
    keypoints2 = (keypoints2 - min2) / (max2 - min2 + eps) * (max1 - min1) + min1

    return np.linalg.norm(keypoints1 - keypoints2)

def oks(gt: np.ndarray, preds: np.ndarray) -> float:
    """Calculate Object Keypoint Similarity."""
    SELECTED_SIGMAS = np.array([
        0.026,  # nose
        0.035, 0.035,  # ears
        0.079, 0.079,  # shoulders
        0.072, 0.072,  # elbows
        0.062, 0.062,  # wrists
        0.107, 0.107,  # hips
        0.087, 0.087,  # knees
        0.089, 0.089,  # ankles
        0.089, 0.089,  # heels
        0.072, 0.072   # foot indices
    ])
    distance = np.linalg.norm(gt - preds, axis=1)
    kp_c = SELECTED_SIGMAS * 2
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

async def get_scores_service(folder_id: str):
    root_path = os.path.join("data", folder_id)
    target_path = os.path.join(root_path, FilePaths.ORIGIN_H5.value)
    user_path = os.path.join(root_path, FilePaths.USER_H5.value)

    _, _, all_frame_points1 = read_pose(target_path)
    _, _, all_frame_points2 = read_pose(user_path)

    score = calculate_score(all_frame_points1, all_frame_points2)
    
    return JSONResponse(content={"score": score}, status_code=200)