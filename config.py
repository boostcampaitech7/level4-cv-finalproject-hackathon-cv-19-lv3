#########################################################################
# !warning : 해당 파일에서 각종 상수값들을 관리합니다. 
# SELECTED_KEYPOINTS dictionary를 제외한 다른 변수들의 값을 변경하실 경우
# 전체 코드가 제대로 동작하지 않을 가능성이 있습니다.
#########################################################################
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

REVERSE_KEYPOINT_MAPPING = {
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

WEIGHTS = {
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
TOTAL_KEYPOINTS = [i for i in KEYPOINT_MAPPING.keys()]
NORMALIZED_LANDMARK_KEYS = ['x', 'y', 'z', 'visibility', 'presense']




#######################################################################################
SELECTED_KEYPOINTS = TOTAL_KEYPOINTS # 사용할 키포인트 지정
SELECTED_KEYPOINTS_MAPPING = {KEYPOINT_MAPPING[SELECTED_KEYPOINTS[i]]: i for i in range(len(SELECTED_KEYPOINTS))}
SELECTED_SIGMAS = [K_I_VALUE[k] for k in SELECTED_KEYPOINTS_MAPPING.keys()]
TOTAL_SIGMAS = [v for v in K_I_VALUE.values()]


#######################################################################################
SEPARATOR = ' ' # dictionary key값에서 단어를 이을 때 어떤 걸 쓸지. '_'또는 ' '