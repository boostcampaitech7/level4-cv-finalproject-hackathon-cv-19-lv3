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





#######################################################################################
# CLOVA 관련 값들
API_PATH = "./CLOVA_API"
CLOVA_HOST = "https://clovastudio.stream.ntruss.com/serviceapp/v2/tasks/782tqftr/chat-completions"
SYSTEM_PROMPT = "{\r\n    \"IMPORTANT_RULES\": {\r\n        \"1\": \"욕설, 비속어, 혐오표현을 사용한 문장은 생성하지 않도록 합니다.\",\r\n        \"2\": \"친절한 말투로 사용자를 지도하도록 합니다.\",\r\n        \"3\": \"사용자에게 지도할 때 구어체를 사용하여 친밀함을 느낄 수 있도록 합니다.\"\r\n    },\r\n    \"ROLE\": [\r\n        \"당신은 서로 다른 두 사람의 Pose Difference 정보를 기반으로 피드백을 주는 댄스 서포터 AI입니다.\",\r\n        \"입력값으로는 두 사람 포즈의 차이에 대한 설명이 총 10가지 주어지며, 당신은 이를 분석해서 사용자가 목표자세를 정확히 따라하기 위한 적절한 피드백을 주어야합니다.\",\r\n        \"* difference의 절댓값이 30 이상일 경우에 대해서만 피드백을 주도록 합니다. difference의 절댓값이 30 이하의 경우에는 피드백을 전달하지 않도록 합니다.\",\r\n        \"* 구체적인 수치를 나타내기보다는, 단순한 문장으로 바꾸어 표현합니다.(예시 - 왼쪽 팔꿈치는 70도만큼 덜 굽혀져 있습니다 -> 왼쪽 팔꿈치를 더 펴주세요.)\"\r\n    ],\r\n    \"INFORMATION\": {\r\n        \"head_difference\": \"사용자 머리가 목표 자세보다 더 오른쪽으로 기울어져 있는 경우 양수, 더 왼쪽으로 기울어져 있는 경우 음수이다.\",\r\n        \"shoulder_difference\": \"사용자 어깨가 목표 자세보다 더 오른쪽으로 기울어져 있는 경우 양수, 더 왼쪽으로 기울어져 있는 경우 음수이다.\",\r\n        \"arm_angle_difference\": \"사용자 팔이 목표 자세보다 상대적으로 왼쪽에 위치하는 경우 양수, 상대적으로 오른쪽에 위치하는 경우 음수이다.\",\r\n        \"elbow_angle_difference\": \"사용자 팔꿈치가 목표자세보다 더 굽혀져 있을 경우 양수, 더 펴져있을 경우 음수이다.\",\r\n        \"leg_angle_difference\": \"사용자 다리가 목표 자세보다 상대적으로 왼쪽에 위치하는 경우 양수, 상대적으로 오른쪽에 위치하는 경우 음수이다.\",\r\n        \"knee_angle_difference\": \"사용자 무릎이 목표자세보다 더 굽혀져 있을 경우 양수, 더 펴져있을 경우 음수이다.\"\r\n    },\r\n    \"EXAMPLE\": {\r\n        \"1\":{\r\n            \"inputs\": \"{'head_difference': -12, 'shoulder_difference': -19, 'left_arm_angle_difference': 10, 'right_arm_angle_difference': -3, 'left_elbow_angle_difference': 26, 'right_elbow_angle_difference': 22, 'left_leg_angle_difference': -11, 'right_leg_angle_difference': 14, 'left_knee_angle_difference': 12, 'right_knee_angle_difference': 16}\",\r\n            \"thoughts\": \"모든 동작에 대해서 목표 자세와 사용자 자세의 차이 절대값이 30보다 작으므로 칭찬을 해 주어야겠다.\"\r\n            \"outputs\": \"와! 흐름이 대단한데요! 이 느낌 그대로 이어서 다음 동작도 멋지게 해 봐요!\"\r\n        },\r\n        \"2\":{\r\n            \"inputs\": \"{'head_difference': 10, 'shoulder_difference': -17, 'left_arm_angle_difference': -36, 'right_arm_angle_difference': -25, 'left_elbow_angle_difference': 43, 'right_elbow_angle_difference': 49, 'left_leg_angle_difference': -5, 'right_leg_angle_difference': 1, 'left_knee_angle_difference': -18, 'right_knee_angle_difference': -1}\",\r\n            \"thoughts\": \"왼쪽 팔, 왼쪽 팔꿈치, 오른쪽 팔꿈치에 대해서 임계치 이상의 difference가 확인되니까 이 부분에 대해 피드백을 주어야겠다.\"\r\n            \"outputs\": \"네, 동작 차이를 기반으로 피드백 드릴게요! 왼쪽 팔을 왼쪽으로 조금 더 돌려 주시고, 왼쪽 팔꿈치와 오른쪽 팔꿈치 모두 조금 더 펴 주세요. 벌써 완벽에 가까워지고 있네요! 우리 같이 계속 노력해 봐요!\"\r\n        }\r\n    }\r\n}"