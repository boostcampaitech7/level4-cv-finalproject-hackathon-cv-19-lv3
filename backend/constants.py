from enum import Enum

class FilePaths(Enum):
    ORIGIN_H5 = "video.h5"
    USER_H5 = "user.h5"
    ORIGIN_MP4 = "video.mp4"
    USER_MP4 = "user.mp4"

class ResponseMessages(Enum):
    FEEDBACK_POSE_FAIL = "포즈가 감지되지 않아 피드백을 드릴 수 없습니다 (┬┬﹏┬┬)"
    H5FILE_LOAD_FAIL = "Failed to read pose data from {file}: {error}"
    FEEDBACK_CLEAR_SUCCESS = "Cache cleared for folder_id: {folder_id}"
    POSE_EXTRACT_POSE_SUCCESS = "Success video pose extract"

NUM_CLASSES = 33
KEYPOINT_MAPPING = {
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

KEYPOINTS_WEIGHTS = {
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

K_I_VALUE = {
    'nose': 0.026,
    'left_eye_inner': 0.025,
    'left_eye': 0.025,
    'left_eye_outer': 0.025,
    'right_eye_inner': 0.025,
    'right_eye': 0.025,
    'right_eye_outer': 0.025,
    'left_ear': 0.035,
    'right_ear': 0.035,
    'mouth_left': 0.026,
    'mouth_right': 0.026,
    'left_shoulder': 0.079,
    'right_shoulder': 0.079,
    'left_elbow': 0.072,
    'right_elbow': 0.072,
    'left_wrist': 0.062,
    'right_wrist': 0.062,
    'left_pinky': 0.072,
    'right_pinky': 0.072,
    'left_index': 0.072,
    'right_index': 0.072,
    'left_thumb': 0.072,
    'right_thumb': 0.072,
    'left_hip': 0.107,
    'right_hip': 0.107,
    'left_knee': 0.087,
    'right_knee': 0.087,
    'left_ankle': 0.089,
    'right_ankle': 0.089,
    'left_heel': 0.089,
    'right_heel': 0.089,
    'left_foot_index': 0.072,
    'right_foot_index': 0.072
}
# select for oks calculation
TOTAL_KEYPOINTS = list(KEYPOINT_MAPPING.keys())
REVERSE_KEYPOINT_MAPPING = {v: k for k, v in KEYPOINT_MAPPING.items()}
NORMALIZED_LANDMARK_KEYS = ['x', 'y', 'z', 'visibility', 'presense']
SELECTED_KEYPOINTS = TOTAL_KEYPOINTS
SELECTED_KEYPOINTS_MAPPING = {KEYPOINT_MAPPING[SELECTED_KEYPOINTS[i]]: i for i in range(len(SELECTED_KEYPOINTS))}
SELECTED_SIGMAS = [K_I_VALUE[k] for k in SELECTED_KEYPOINTS_MAPPING.keys()]
TOTAL_SIGMAS = [v for v in K_I_VALUE.values()]

body_parts_korean = {
    "nose": "코",
    "left_eye_inner": "왼쪽 눈 안쪽",
    "left_eye": "왼쪽 눈",
    "left_eye_outer": "왼쪽 눈 바깥쪽",
    "right_eye_inner": "오른쪽 눈 안쪽",
    "right_eye": "오른쪽 눈",
    "right_eye_outer": "오른쪽 눈 바깥쪽",
    "left_ear": "왼쪽 귀",
    "right_ear": "오른쪽 귀",
    "mouth_left": "왼쪽 입",
    "mouth_right": "오른쪽 입",
    "left_shoulder": "왼쪽 어깨",
    "right_shoulder": "오른쪽 어깨",
    "left_elbow": "왼쪽 팔꿈치",
    "right_elbow": "오른쪽 팔꿈치",
    "left_wrist": "왼쪽 손목",
    "right_wrist": "오른쪽 손목",
    "left_pinky": "왼쪽 새끼손가락",
    "right_pinky": "오른쪽 새끼손가락",
    "left_index": "왼쪽 검지",
    "right_index": "오른쪽 검지",
    "left_thumb": "왼쪽 엄지",
    "right_thumb": "오른쪽 엄지",
    "left_hip": "왼쪽 골반",
    "right_hip": "오른쪽 골반",
    "left_knee": "왼쪽 무릎",
    "right_knee": "오른쪽 무릎",
    "left_ankle": "왼쪽 발목",
    "right_ankle": "오른쪽 발목",
    "left_heel": "왼쪽 발뒤꿈치",
    "right_heel": "오른쪽 발뒤꿈치",
    "left_foot_index": "왼쪽 발가락",
    "right_foot_index": "오른쪽 발가락",
    "eye": "눈",
    "ear": "귀",
    "mouth": "입",
    "shoulder": "어깨",
    "elbow": "팔꿈치",
    "wrist": "손목",
    "pinky": "새끼손가락",
    "index": "검지",
    "thumb": "엄지",
    "hip": "엉덩이",
    "knee": "무릎",
    "ankle": "발목",
    "heel": "발뒤꿈치",
    "foot_index": "발가락",
    "left_hand": "왼손",
    "right_hand": "오른손",
    "hand": "손",
    "right_foot": "오른발",
    "left_foot": "왼발",
    "foot": "발",
    "torso": "몸",
    'waist': "허리",
    "left_waist": "왼쪽 허리",
    "right_waist": "오른쪽 허리",
    "arm": "팔",
    "left_arm": "왼팔",
    "right_arm": "오른팔",
    "leg": "다리",
    "left_leg": "왼쪽 다리",
    "right_leg": "오른쪽 다리",
    "belly": "배",
    "head": "머리",
    "breast": "가슴"
}

feature_types = {
    'head': {
        'lower_angle_difference': 'int',
        'direction_difference': 'int'
    },
    'body': {
        'bend_angle_difference': 'int',
        'direction_difference': 'int'
    },
    'left_arm': {
        'bend_angle_difference': 'int',
        'arm_height_difference': 'int',
        'hand_height_difference': 'int',
        'direction_difference': 'int',
        'closest_point_difference': {
            'pose1': 'float',
            'pose2': 'float',
            'diff': 'list'
        }
    },
    'right_arm': {
        'bend_angle_difference': 'int',
        'arm_height_difference': 'int',
        'hand_height_difference': 'int',
        'direction_difference': 'int',
        'closest_point_difference': {
            'pose1': 'float',
            'pose2': 'float',
            'diff': 'list'
        }
    },
    'left_leg': {
        'bend_angle_difference': 'int',
        'height_difference': 'int',
        'direction_difference': 'int'
    },
    'right_leg': {
        'bend_angle_difference': 'int',
        'height_difference': 'int',
        'direction_difference': 'int'
    },
    'leg': {
        'knee_distance_difference': 'float',
        'foot_distance_difference': 'float'
    }
}
