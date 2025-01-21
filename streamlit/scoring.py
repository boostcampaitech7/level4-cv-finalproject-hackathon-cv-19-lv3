from keypoint_map import KEYPOINT_MAPPING, SELECTED_KEYPOINTS, SELECTED_SIGMAS, SELECTED_KEYPOINTS_MAPPING
import numpy as np
import cv2

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

def normalize_landmarks_to_range(landmarks_np_1, landmarks_np_2):
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
    normalized_landmarks2 = (landmarks_np_2[:, :3] - min2) / (max2 - min2) * (max1 - min1) + min1

    # Combine normalized coordinates with the original visibility values
    normalized_landmarks2 = np.hstack((normalized_landmarks2, landmarks_np_2[:, 3:4]))

    return normalized_landmarks2

def normalize_landmarks_to_range_by_mean(all_landmarks_np_1, all_landmarks_np_2):
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
    normalized_landmarks2 = (all_landmarks_np_2[:, :, :3] - min2_mean) / (max2_mean - min2_mean) * (max1_mean - min1_mean) + min1_mean

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


def evaluate_everything(landmarks1_np, bs1, landmarks2_np, bs2, pck_thres=0.1, normalize=True, verbose=True, ignore_z=False):
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


def main(p1, p2):
    from detector import PoseDetector
    import matplotlib.pyplot as plt
    d = PoseDetector()

    l1, seg1, ann_img1, bs1 = d.get_detection(p1)
    l2, seg2, ann_img2, bs2 = d.get_detection(p2)
    np_l1 = refine_landmarks(l1)
    np_l2 = refine_landmarks(l2)
    evaluate_everything(np_l1, bs1, np_l2, bs2, ignore_z=True)
    

if __name__=="__main__":
    img_path1 = "images/jun_v.jpg"
    img_path2 = "images/wrong_pose_img.jpg"
    main(img_path1, img_path2)