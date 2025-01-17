from keypoint_map import REVERSE_KEYPOINT_MAPPING, KEYPOINT_MAPPING, SELECTED
import numpy as np

# pos normalization function
def normalize_landmarks(landmarks_np, mode="default", root_idx1=None, root_idx2=None, norm_idx1=None, norm_idx2=None):
    if mode == "shoulder":
        right_shoulder_idx = norm_idx1 if norm_idx1 else REVERSE_KEYPOINT_MAPPING['right_shoulder']
        left_shoulder_idx = norm_idx2 if norm_idx2 else REVERSE_KEYPOINT_MAPPING['left_shoulder']
        d = np.linalg.norm(landmarks_np[right_shoulder_idx, :3] - landmarks_np[left_shoulder_idx, :3])
    elif mode == 'l2':
        d = np.linalg.norm(landmarks_np[..., :3].flatten())
    else:
        right_hip_idx = root_idx1 if root_idx1 else REVERSE_KEYPOINT_MAPPING['right_hip']
        left_hip_idx = root_idx2 if root_idx2 else REVERSE_KEYPOINT_MAPPING['left_hip']
        hip_center = (landmarks_np[right_hip_idx, :3] + landmarks_np[left_hip_idx, :3]) / 2.
        landmarks_np[..., :3] = landmarks_np[..., :3] - hip_center
        right_shoulder_idx = norm_idx1 if norm_idx1 else REVERSE_KEYPOINT_MAPPING['right_shoulder']
        left_shoulder_idx = norm_idx2 if norm_idx2 else REVERSE_KEYPOINT_MAPPING['left_shoulder']
        d = np.linalg.norm((landmarks_np[right_shoulder_idx, :3] - landmarks_np[left_shoulder_idx, :3]))

    landmarks_np[..., :3] = landmarks_np[..., :3] / d
    return landmarks_np / d

# select only important feature in all keypoint
def filter_important_features(landmarks_np, targets=SELECTED):
    return landmarks_np[targets]


def cos_sim(landmarks1, landmarks2):
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("both landmarks must have same shape!!")
    
    if landmarks1.shape[-1] == 4:
        landmarks1 = landmarks1[..., :3]
        landmarks2 = landmarks2[..., :3]
    landmarks1 = landmarks1.flatten()
    landmarks2 = landmarks2.flatten()
    return (1+np.dot(landmarks1, landmarks2)/(np.linalg.norm(landmarks1)*np.linalg.norm(landmarks2))) / 2


def refine_landmarks(landmarks):
    lst = []
    for landmark in landmarks:
        lst.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(lst)