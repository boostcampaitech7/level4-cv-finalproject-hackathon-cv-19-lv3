import cv2
import math
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Mediapipe 랜드마크 이름
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", 
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", 
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", 
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
    "right_heel", "left_foot_index", "right_foot_index"
]

def extract_keypoints_to_list(input_file):
    # 이미지 읽기
    input_img = cv2.imread(input_file)
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Pose 추출
    results = pose.process(input_img_rgb)

    keypoints = []
    if results.pose_landmarks:
        # 키포인트 데이터를 (부위 이름, x, y, z, visibility)로 저장
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append((
                # landmark_names[id],  # 부위 이름
                id,                  # 부위 인덱스
                landmark.x,          # x 좌표
                landmark.y,          # y 좌표
                landmark.z,          # z 좌표
                landmark.visibility  # 가시성
            ))
        print("Keypoints extracted:")
        print(keypoints)
    else:
        print("No pose landmarks detected.")
    
    return keypoints

def normalize_two_keypoints(keypoints1, keypoints2, reference_indices):
    """
    두 개의 키포인트 데이터를 공통 기준점과 스케일로 정규화.
    
    Args:
        keypoints1 (list): 첫 번째 Mediapipe 키포인트 데이터 (x, y, z, visibility 포함).
        keypoints2 (list): 두 번째 Mediapipe 키포인트 데이터.
        reference_indices (list): 기준점을 계산할 랜드마크 인덱스 리스트.
        
    Returns:
        tuple: (정규화된 keypoints1, 정규화된 keypoints2)
    """
    def calculate_reference_point(keypoints, indices):
        # 기준점(중심 좌표) 계산
        ref_x = sum(keypoints[i][1] for i in indices) / len(indices)
        ref_y = sum(keypoints[i][2] for i in indices) / len(indices)
        return ref_x, ref_y

    def scale_keypoints(keypoints, scale_factor):
        # 모든 키포인트를 스케일링
        return [(kp[0] / scale_factor, kp[1] / scale_factor, kp[2] / scale_factor, kp[3]) for kp in keypoints]

    # 기준점 계산
    ref1_x, ref1_y = calculate_reference_point(keypoints1, reference_indices)
    ref2_x, ref2_y = calculate_reference_point(keypoints2, reference_indices)
    
    # 각 데이터의 기준점 이동
    normalized_kp1 = [(kp[0] - ref1_x, kp[1] - ref1_y, kp[2], kp[3]) for kp in keypoints1]
    normalized_kp2 = [(kp[0] - ref2_x, kp[1] - ref2_y, kp[2], kp[3]) for kp in keypoints2]
    
    # 기준 길이(스케일) 계산 (예: 어깨 너비)
    def calculate_scale(keypoints, indices):
        x1, y1 = keypoints[indices[0]][0], keypoints[indices[0]][1]
        x2, y2 = keypoints[indices[1]][0], keypoints[indices[1]][1]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    scale1 = calculate_scale(normalized_kp1, reference_indices)
    scale2 = calculate_scale(normalized_kp2, reference_indices)
    
    # 공통 스케일로 맞춤 (평균 스케일 사용)
    avg_scale = (scale1 + scale2) / 2.0
    scaled_kp1 = scale_keypoints(normalized_kp1, avg_scale)
    scaled_kp2 = scale_keypoints(normalized_kp2, avg_scale)
    
    return scaled_kp1, scaled_kp2

# 이미지 파일 경로
input_image_path1 = "img_file/1.png"
input_image_path2 = "img_file/2.png"

# 함수 실행 -> (부위 이름(또는 인덱스), x, y, z, visibility) 형태
keypoints1 = extract_keypoints_to_list(input_image_path1)
keypoints2 = extract_keypoints_to_list(input_image_path2)

