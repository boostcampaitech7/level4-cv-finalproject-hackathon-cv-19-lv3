import cv2
import json
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result, landmark_color, connection_color):
    pose_landmarks_list = detection_result.pose_landmarks
    
    # 랜드마크 그리기를 위한 DrawingSpec 설정
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # 스타일 설정
    landmark_drawing_spec = mp_drawing.DrawingSpec(
        color=landmark_color,
        thickness=10,
        circle_radius=10
    )
    connection_drawing_spec = mp_drawing.DrawingSpec(
        color=connection_color,
        thickness=10
    )

    # 각 포즈의 랜드마크 그리기
    for pose_landmarks in pose_landmarks_list:
        # 랜드마크를 정규화된 프로토콜 버퍼에서 이미지 좌표로 변환
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])

        # 랜드마크와 연결선 그리기
        mp_drawing.draw_landmarks(
            image=rgb_image,
            landmark_list=pose_landmarks_proto,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
    
    return rgb_image

def extract_pose_landmarks(result):
    landmarks = {}
    
    # 각 포즈의 첫 번째 검출 결과만 사용
    if result.pose_landmarks:
        pose_landmarks = result.pose_landmarks[0]
        
        # 각 부위별 랜드마크 좌표 추출
        landmarks = {
            "face": {
                "0": {"x": pose_landmarks[0].x, "y": pose_landmarks[0].y, "z": pose_landmarks[0].z},
                "7": {"x": pose_landmarks[7].x, "y": pose_landmarks[7].y, "z": pose_landmarks[7].z},
                "8": {"x": pose_landmarks[8].x, "y": pose_landmarks[8].y, "z": pose_landmarks[8].z}
            },
            "left_arm": {
                "11": {"x": pose_landmarks[11].x, "y": pose_landmarks[11].y, "z": pose_landmarks[11].z},
                "13": {"x": pose_landmarks[13].x, "y": pose_landmarks[13].y, "z": pose_landmarks[13].z},
                "15": {"x": pose_landmarks[15].x, "y": pose_landmarks[15].y, "z": pose_landmarks[15].z}
            },
            "right_arm": {
                "12": {"x": pose_landmarks[12].x, "y": pose_landmarks[12].y, "z": pose_landmarks[12].z},
                "14": {"x": pose_landmarks[14].x, "y": pose_landmarks[14].y, "z": pose_landmarks[14].z},
                "16": {"x": pose_landmarks[16].x, "y": pose_landmarks[16].y, "z": pose_landmarks[16].z}
            },
            "left_leg": {
                "23": {"x": pose_landmarks[23].x, "y": pose_landmarks[23].y, "z": pose_landmarks[23].z},
                "25": {"x": pose_landmarks[25].x, "y": pose_landmarks[25].y, "z": pose_landmarks[25].z},
                "27": {"x": pose_landmarks[27].x, "y": pose_landmarks[27].y, "z": pose_landmarks[27].z}
            },
            "right_leg": {
                "24": {"x": pose_landmarks[24].x, "y": pose_landmarks[24].y, "z": pose_landmarks[24].z},
                "26": {"x": pose_landmarks[26].x, "y": pose_landmarks[26].y, "z": pose_landmarks[26].z},
                "28": {"x": pose_landmarks[28].x, "y": pose_landmarks[28].y, "z": pose_landmarks[28].z}
            },
            "left_foot": {
                "27": {"x": pose_landmarks[27].x, "y": pose_landmarks[27].y, "z": pose_landmarks[27].z},
                "29": {"x": pose_landmarks[29].x, "y": pose_landmarks[29].y, "z": pose_landmarks[29].z},
                "31": {"x": pose_landmarks[31].x, "y": pose_landmarks[31].y, "z": pose_landmarks[31].z}
            },
            "right_foot": {
                "28": {"x": pose_landmarks[28].x, "y": pose_landmarks[28].y, "z": pose_landmarks[28].z},
                "30": {"x": pose_landmarks[30].x, "y": pose_landmarks[30].y, "z": pose_landmarks[30].z},
                "32": {"x": pose_landmarks[32].x, "y": pose_landmarks[32].y, "z": pose_landmarks[32].z}
            }
        }
    
    return landmarks

input_pose1 = "../img_file/stand.png"
input_pose2 = "../img_file/comp1.png"

output_pose1 = "stand.json"
output_pose2 = "comp1.json"

# 이미지 읽기
image1 = cv2.imread(input_pose1)
image2 = cv2.imread(input_pose2)

image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=image1_rgb)
mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=image2_rgb)

# Mediapipe PoseLandmarker 설정
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

result1 = detector.detect(mp_image1)
result2 = detector.detect(mp_image2)

# 두 포즈의 랜드마크 추출
pose1_landmarks = extract_pose_landmarks(result1)
pose2_landmarks = extract_pose_landmarks(result2)

# JSON 파일로 저장
with open(output_pose1, 'w') as f:
    json.dump(pose1_landmarks, f, indent=4)

with open(output_pose2, 'w') as f:
    json.dump(pose2_landmarks, f, indent=4)

# 첫 번째 포즈는 빨간색 계열로 시각화
output_image = draw_landmarks_on_image(
    image1_rgb.copy(), 
    result1, 
    landmark_color=(255, 0, 0),    # 빨간색 랜드마크
    connection_color=(255, 150, 150)  # 연한 빨간색 연결선
)

# 두 번째 포즈는 파란색 계열로 시각화 (같은 캔버스에 그리기)
output_image = draw_landmarks_on_image(
    output_image, 
    result2, 
    landmark_color=(0, 0, 255),    # 파란색 랜드마크
    connection_color=(150, 150, 255)  # 연한 파란색 연결선
)

# RGB에서 BGR로 변환하여 저장/표시
final_output = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# 결과 저장
cv2.imwrite('overlaid_poses.png', final_output)

# 결과 표시
cv2.imshow('Overlaid Poses', final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()