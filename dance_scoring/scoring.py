import sys
sys.path.append("./")
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw
from config import KEYPOINT_MAPPING, SELECTED_KEYPOINTS, SELECTED_SIGMAS


# refine landmark result to numpy array
def refine_landmarks(landmarks, target_keys=None):
    """
    Pose Object로 구성된 list를 numpy array로 변환하여 반환
    [PoseObject1(x, y, z, v, p), ...] -> numpy array
    """
    if target_keys is None:
        target_keys = SELECTED_KEYPOINTS

    lst = []
    for i, landmark in enumerate(landmarks):
        if i in target_keys:
            lst.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(lst)

# select only important feature in all keypoint
def filter_important_features(landmarks_np, targets=SELECTED_KEYPOINTS):
    return landmarks_np[targets]


def get_center_pair_frames(pairs, keypoint2_index, matched_idx=1):
    # keypoint2_index와 매칭된 keypoint1의 모든 인덱스를 찾음
    matching_pairs = [pair for pair in pairs if pair[matched_idx] == keypoint2_index]
    num_total_pairs = len(matching_pairs)

    if not matching_pairs:
        print(f"No matching pairs found for keypoint2 index: {keypoint2_index}")
        return (0, 0)

    # 매칭된 프레임 중 시간축 기준 중간에 위치하는 것을 추출
    return matching_pairs[num_total_pairs // 2][not matched_idx]

def get_all_pair_frames(pairs, keypoint2_index, matched_idx=1):
    # keypoint2_index와 매칭된 keypoint1의 모든 인덱스를 찾음
    matching_pairs = [pair for pair in pairs if pair[matched_idx] == keypoint2_index]

    if not matching_pairs:
        print(f"No matching pairs found for keypoint2 index: {keypoint2_index}")
        return (0, 0)

    # keypoint1 매칭된 프레임 번호 모두 return
    return matching_pairs


def normalize_landmarks(landmarks_np, box):
    """
    Normalize landmarks based on the bounding box dimensions.
    
    Parameters:
        landmarks (numpy array): Keypoints array of shape (num_selected_point, 4) (x, y, z, visibility).
        box (numpy array): Bounding box dimensions [width, height, depth].
        
    Returns:
        numpy array: Normalized landmarks of shape (num_selected_point, 4).
    """
    normalized_landmarks = np.copy(landmarks_np)
    normalized_landmarks[:, :3] /= box  # Normalize x, y, z by dividing by bounding box dimensions
    return normalized_landmarks

def normalize_landmarks_to_range(landmarks_np_1, landmarks_np_2, eps=1e-7):
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



def cos_sim(landmarks1, landmarks2, ignore_z = False):
    target_end = 3-ignore_z

    if landmarks1.shape != landmarks2.shape:
        raise ValueError("both landmarks must have same shape!!")
    
    landmarks1 = landmarks1[..., :target_end]
    landmarks2 = landmarks2[..., :target_end]
    
    cos_score = 0.
    for i in range(target_end):
        cos_score += (1+np.dot(landmarks1[..., i], landmarks2[..., i])/(np.linalg.norm(landmarks1[..., i])*np.linalg.norm(landmarks2[..., i]))) / 2
    return cos_score / target_end


def L1_score(landmarks1, landmarks2, ignore_z = False):
    if landmarks1 is None or landmarks2 is None:
        return 0  # 비교 불가 시 유사성 0으로 처리
    
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("both landmarks must have same shape!!")
    target_end = 3-ignore_z
    if landmarks1.shape[-1] == 4:
        landmarks1 = landmarks1[..., :target_end]
        landmarks2 = landmarks2[..., :target_end]
    # 평준화된 유클리드 거리 계산
    distance = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    similarity = 1 / (1 + np.mean(distance))  # 유사성을 0~1로 정규화
    return similarity

# 단순히 scipy의 euclidean, cosine함수 썼을 때 범위 맞춰주는 함수들
def make_cosine_similarity(avg_cosine):
    # cos = (avg_cosine * 1000) % 100
    cos = (1 + avg_cosine) / 2 * 100
    return cos

def make_euclidean_similarity(avg_euclidean):
    euc = 100 - (avg_euclidean * 100)
    return euc

# OKS 값 계산 함수
def oks(gt, preds, boxsize, ignore_z=False):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    target_end = 3-ignore_z
    sigmas = np.array(SELECTED_SIGMAS)
    distance = np.linalg.norm(gt[:, :target_end] - preds[:, :target_end], axis=1)

    if ignore_z:
        bbox_gt = boxsize[0] ** 2 + boxsize[1] ** 2
    else:
        bbox_gt = boxsize[0] ** 2 + boxsize[1] ** 2 + boxsize[2] ** 2
    kp_c = sigmas * 2
    return np.mean(np.exp(-(distance ** 2) / (2 * (bbox_gt) * (kp_c ** 2))))


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


def evaluate_on_two_images(landmarks1_np, bs1, landmarks2_np, pck_thres=0.1, normalize=True, verbose=True, ignore_z=False):
    """
    landmarks1_np, landmarks2_np: [num_selected_keypoints, 4]의 numpy array
    pck_thres : pck스코어 계산 시 얼마나 가까워야 match시킬 것인지
    normalize: minmax 정규화 적용 여부
    verbose: results 결과 print여부
    ignore_z: z값을 무시하고 metric을 계산할지 여부
    """
    target_end = 3-ignore_z
    if normalize:
        l2 = normalize_landmarks_to_range(landmarks1_np, landmarks2_np)
    else:
        l2 = landmarks2_np
    l1 = landmarks1_np

    pck_score, matched = pck(l1, l2, pck_thres, ignore_z)
    results = {
        "L1_score":  L1_score(l1, l2, ignore_z),
        "L2_distance": np.linalg.norm((l1 - l2)[..., :target_end].flatten()),
        "cos_similarity": cos_sim(l1, l2, ignore_z),
        f"PCK(thres={pck_thres:.2f})": pck_score,
        "oks:": oks(l1, l2, bs1, ignore_z),
        "matched": {
            KEYPOINT_MAPPING[real_idx]:matched[i] for i, real_idx in enumerate(SELECTED_KEYPOINTS)
        }
    }

    if verbose:
        for k, v in results.items():
            print(f"{k}: {v}")
    return results


def evaluate_on_two_videos(all_landmarks1, all_landmarks2, score_target='PCK', pck_thres=0.1, thres=0.4, ignore_z=False, use_dtw=False):
    """
    all_landmarks1, all_landmarks2: list[landmarks]
    pck_thres : pck스코어 계산 시 얼마나 가까워야 match시킬 것인지
    thres : score target으로 선정된 metric 값이 해당 thres 이하면 low-score_frames로 분류
    ignore_z: z값을 무시하고 metric을 계산할지 여부
    use_dtw: True의 경우 fastdtw를 통해 frame을 매칭해서 스코어를 계산. False의 경우 짧은 영상 기준으로 긴 영상을 자름

    returns:
        total_results: 각 metric별 score를 담은 dictionary
        low_score_frames: all_landmarks2기준으로 all_landmarks1과 비교에서 낮은 스코어를 기록한 frame number list
    """
    total_results = defaultdict(list)
    low_score_frames = []
    bs1 = np.array([1, 0, 0])

    if use_dtw:
        all_landmarks_np_1 = np.array([refine_landmarks(l) for l in all_landmarks1])
        all_landmarks_np_2 = np.array([refine_landmarks(l) for l in all_landmarks2])
        all_landmarks_np_2 = normalize_landmarks_to_range_by_mean(all_landmarks_np_1, all_landmarks_np_2)

        _, path = fastdtw(
            all_landmarks_np_1[..., :3].reshape(all_landmarks_np_1.shape[0], -1),
            all_landmarks_np_2[..., :3].reshape(all_landmarks_np_2.shape[0], -1),
            dist=euclidean
        )

        for frame_num_1, frame_num_2 in path:
            np_l1 = all_landmarks_np_1[frame_num_1]
            np_l2 = all_landmarks_np_2[frame_num_2]
            results = evaluate_on_two_images(np_l1, bs1, np_l2, pck_thres=pck_thres, verbose=False, ignore_z=ignore_z)
            results['matched_frame'] = (frame_num_1, frame_num_2)

            for k, v in results.items():
                total_results[k].append(v)
                if score_target in k and results[k] < thres:
                    low_score_frames.append(frame_num_2)
    else:
        for frame_num, (landmarks1, landmarks2) in enumerate(zip(all_landmarks1, all_landmarks2)):
            np_l1 = refine_landmarks(landmarks1)
            np_l2 = refine_landmarks(landmarks2)
            results = evaluate_on_two_images(np_l1, bs1, np_l2, pck_thres=pck_thres, verbose=False, ignore_z=ignore_z)
            results['matched_frame'] = (frame_num, frame_num)

            for k, v in results.items():
                total_results[k].append(v)
                if score_target in k and results[k] < thres:
                    low_score_frames.append(frame_num)
    
    for k in total_results.keys():
        if "matched" in k:
            continue
        total_results[k] = np.mean(total_results[k])
    
    return total_results, low_score_frames


def normalize_landmarks_to_range_dist(keypoints1: np.ndarray, keypoints2: np.ndarray, eps: float = 1e-7) -> float:
    """Normalize pose landmarks origin video frame and user video frame."""
    min1 = np.min(keypoints1, axis=0)
    max1 = np.max(keypoints1, axis=0)
    min2 = np.min(keypoints2, axis=0)
    max2 = np.max(keypoints2, axis=0)

    keypoints1 = keypoints1
    keypoints2 = (keypoints2 - min2) / (max2 - min2 + eps) * (max1 - min1) + min1

    return np.linalg.norm(keypoints1 - keypoints2)


def calculate_similarity(keypoints1: np.ndarray, keypoints2: np.ndarray, pck_threshold=0.1):
    """Calculate similarity between two sequences of keypoints."""
    _, pairs = fastdtw(keypoints1, keypoints2, dist=normalize_landmarks_to_range_dist)

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

        oks_score = oks(kp1, kp2, boxsize=np.array([1,0,0]))
        oks_scores.append(oks_score)

        pck_score, _ = pck(kp1, kp2, threshold=pck_threshold)
        pck_scores.append(pck_score)

    avg_cosine = np.mean(cosine_similarities)
    avg_euclidean = np.mean(euclidean_similarities)
    
    average_oks = np.mean(oks_scores)
    average_pck = np.mean(pck_scores)

    return avg_cosine, avg_euclidean, average_oks, average_pck, pairs

def calculate_score(keypoints1, keypoints2):
    """Calculate final score."""
    results = calculate_similarity(keypoints1, keypoints2)
    avg_cosine, avg_euclidean, average_oks, average_pck, _ = results

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