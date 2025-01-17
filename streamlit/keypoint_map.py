import numpy as np

NUM_CLASSES = 33
keypoint_mapping = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}
SELECTED = [2, 5, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


reverse_keypoint_mapping = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

weights = {
    "nose": 0.05,
    "left_eye_inner": 0.02,
    "left_eye": 0.02,
    "left_eye_outer": 0.02,
    "right_eye_inner": 0.02,
    "right_eye": 0.02,
    "right_eye_outer": 0.02,
    "left_ear": 0.01,
    "right_ear": 0.01,
    "mouth_left": 0.02,
    "mouth_right": 0.02,
    "left_shoulder": 0.06,
    "right_shoulder": 0.06,
    "left_elbow": 0.04,
    "right_elbow": 0.04,
    "left_wrist": 0.03,
    "right_wrist": 0.03,
    "left_pinky": 0.01,
    "right_pinky": 0.01,
    "left_index": 0.01,
    "right_index": 0.01,
    "left_thumb": 0.01,
    "right_thumb": 0.01,
    "left_hip": 0.06,
    "right_hip": 0.06,
    "left_knee": 0.05,
    "right_knee": 0.05,
    "left_ankle": 0.04,
    "right_ankle": 0.04,
    "left_heel": 0.02,
    "right_heel": 0.02,
    "left_foot_index": 0.02,
    "right_foot_index": 0.02
}

sigma_i_value = {
    'nose': 0.026,
    'eye': 0.025,
    'ear': 0.035,
    'mouse': 0.026,
    'shoulder': 0.079,
    'elbow': 0.072,
    'wrist': 0.062,
    'pinky': 0.072,
    'index': 0.072,
    'thumb': 0.072,
    'hip': 0.107,
    'knee': 0.087,
    'ankle': 0.089,
    'heel': 0.089,
    'foot': 0.089
}
k_i_value = {
    k: sigma_i_value[k] for k in sigma_i_value.keys()
}

def normalize_landmarks(landmarks_np, mode="default"):
    if mode == "shoulder":
        right_shoulder_idx = reverse_keypoint_mapping['right_shoulder']
        left_shoulder_idx = reverse_keypoint_mapping['left_shoulder']
        d = np.linalg.norm(landmarks_np[right_shoulder_idx] - landmarks_np[left_shoulder_idx])
    elif mode == 'l2':
        d = np.linalg.norm(landmarks_np.flatten())
    else:
        right_hip_idx = reverse_keypoint_mapping['right_hip']
        left_hip_idx = reverse_keypoint_mapping['left_hip']
        hip_center = (landmarks_np[right_hip_idx] + landmarks_np[left_hip_idx]) / 2.
        landmarks_np = landmarks_np - hip_center
        right_shoulder_idx = reverse_keypoint_mapping['right_shoulder']
        left_shoulder_idx = reverse_keypoint_mapping['left_shoulder']
        d = np.linalg.norm((landmarks_np[right_shoulder_idx] - landmarks_np[left_shoulder_idx]))

    return landmarks_np / d

def filter_important_features(landmarks_np, targets=SELECTED):
    return landmarks_np[targets]


def landmarks_to_dict(all_landmarks):
    landmark_dict = {}
    
    for i, landmarks in enumerate(all_landmarks):
        d = {j: {
                "name": keypoint_mapping[j],
                "x": landmarks[j][0],
                "y": landmarks[j][1],
                "z": landmarks[j][2],
                "visibility": landmarks[j][3]
            } for j in keypoint_mapping.keys()}
        landmark_dict[i] = d
    return landmark_dict
        