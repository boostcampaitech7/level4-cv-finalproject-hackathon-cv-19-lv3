from keypoint_map import REVERSE_KEYPOINT_MAPPING, SELECTED_KEYPOINTS, SELECTED_SIGMAS, SELECTED_KEYPOINTS_MAPPING
import numpy as np
import cv2

# refine landmark result to numpy array
def refine_landmarks(landmarks):
    lst = []
    for i, landmark in enumerate(landmarks):
        if i in SELECTED_KEYPOINTS:
            lst.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(lst)

# select only important feature in all keypoint
def filter_important_features(landmarks_np, targets=SELECTED_KEYPOINTS):
    return landmarks_np[targets]


def normalize_landmarks(landmarks, box):
    """
    Normalize landmarks based on the bounding box dimensions.
    
    Parameters:
        landmarks (numpy array): Keypoints array of shape (num_selected_point, 4) (x, y, z, visibility).
        box (numpy array): Bounding box dimensions [width, height, depth].
        
    Returns:
        numpy array: Normalized landmarks of shape (num_selected_point, 4).
    """
    normalized_landmarks = np.copy(landmarks)
    normalized_landmarks[:, :3] /= box  # Normalize x, y, z by dividing by bounding box dimensions
    return normalized_landmarks

def normalize_landmarks_to_range(landmarks1, landmarks2):
    """
    Normalize landmarks2 to match the coordinate range of landmarks1.

    Parameters:
        landmarks1 (numpy array): Keypoints array for the first pose (num_selected_point, 4).
        landmarks2 (numpy array): Keypoints array for the second pose (num_selected_point, 4).

    Returns:
        numpy array: Normalized landmarks2 matching the range of landmarks1.
    """
    # Calculate min and max for landmarks1 and landmarks2
    min1 = np.min(landmarks1[:, :3], axis=0)  # (x_min, y_min, z_min) for landmarks1
    max1 = np.max(landmarks1[:, :3], axis=0)  # (x_max, y_max, z_max) for landmarks1

    min2 = np.min(landmarks2[:, :3], axis=0)  # (x_min, y_min, z_min) for landmarks2
    max2 = np.max(landmarks2[:, :3], axis=0)  # (x_max, y_max, z_max) for landmarks2

    # Normalize landmarks2 to the range of landmarks1
    normalized_landmarks2 = (landmarks2[:, :3] - min2) / (max2 - min2) * (max1 - min1) + min1

    # Combine normalized coordinates with the original visibility values
    normalized_landmarks2 = np.hstack((normalized_landmarks2, landmarks2[:, 3:4]))

    return normalized_landmarks2


def cos_sim(landmarks1, landmarks2):
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("both landmarks must have same shape!!")
    
    if landmarks1.shape[-1] == 4:
        landmarks1 = landmarks1[..., :3]
        landmarks2 = landmarks2[..., :3]
    
    cos_score = 0.
    for i in range(3):
        cos_score += (1+np.dot(landmarks1[..., i], landmarks2[..., i])/(np.linalg.norm(landmarks1[..., i])*np.linalg.norm(landmarks2[..., i]))) / 2
    return cos_score / 3


def L1_score(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return 0  # 비교 불가 시 유사성 0으로 처리
    
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("both landmarks must have same shape!!")
    
    if landmarks1.shape[-1] == 4:
        landmarks1 = landmarks1[..., :3]
        landmarks2 = landmarks2[..., :3]
    # 평준화된 유클리드 거리 계산
    distance = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    similarity = 1 / (1 + np.mean(distance))  # 유사성을 0~1로 정규화
    return similarity


# OKS 값 계산 함수
def oks(gt, preds, boxsize):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    sigmas = np.array(SELECTED_SIGMAS)
    distance = np.linalg.norm(gt[:, :3] - preds[:, :3], axis=1)
    bbox_gt = boxsize[0] ** 2 + boxsize[1] ** 2 + boxsize[2] ** 2
    kp_c = sigmas * 2
    return np.mean(np.exp(-(distance ** 2) / (2 * (bbox_gt) * (kp_c ** 2))))


# PCK 값 계산 함수
def pck(gt, preds, threshold):
    """
    gt : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    preds : shape (num_selected, 4) 4 feature is (x, y, z, visibility)
    """
    distance = np.linalg.norm(gt[:, :3] - preds[:, :3], axis=1)
    pck_score = np.mean(distance < threshold)
    return pck_score


def evaluate_everything(landmarks1_np, bs1, landmarks2_np, bs2, pck_thres=0.1):
    l2 = normalize_landmarks_to_range(landmarks1_np, landmarks2_np)
    l1 = landmarks1_np
    print("L1 score : ", L1_score(l1, l2))
    print("L2 distance : ", np.linalg.norm((l1 - l2)[..., :3].flatten()))
    print(f"cos similarity : {cos_sim(l1, l2)}")
    print(f"PCK(thres is {pck_thres}): ", pck(l1, l2, pck_thres))
    print("oks: ", oks(l1, l2, np.array([1,0,0])))


def main(p1, p2):
    from detector import PoseDetector
    import matplotlib.pyplot as plt
    d = PoseDetector()

    l1, seg1, ann_img1, bs1 = d.get_detection(p1)
    l2, seg2, ann_img2, bs2 = d.get_detection(p2)
    np_l1 = refine_landmarks(l1)
    np_l2 = refine_landmarks(l2)
    evaluate_everything(np_l1, bs1, np_l2, bs2)
    


if __name__=="__main__":
    img_path1 = "images/jun_v.jpg"
    img_path2 = "images/wrong_pose_img.jpg"
    main(img_path1, img_path2)