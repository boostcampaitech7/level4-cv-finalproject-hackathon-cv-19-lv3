import cv2
import json
import matplotlib.pyplot as plt
import mediapipe as mp

# Mediapipe의 Pose Connections 불러오기
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

def draw_keypoints_and_connections(input_image_path, input_json_path, circle_color=(0, 255, 0), line_color=(255, 0, 0), 
                                    circle_radius=5, line_thickness=2):
    """
    JSON 파일에서 읽은 키포인트를 이미지에 찍고 연결선을 그려주는 함수.

    Args:
        input_image_path (str): 입력 이미지 경로.
        input_json_path (str): 키포인트 데이터가 포함된 JSON 파일 경로.
        circle_color (tuple): 키포인트의 색상 (BGR).
        line_color (tuple): 연결선의 색상 (BGR).
        circle_radius (int): 키포인트 원의 반지름.
        line_thickness (int): 연결선의 두께.

    Returns:
        None
    """
    # JSON 파일 로드
    with open(input_json_path, "r") as json_file:
        keypoints = json.load(json_file)
    
    # 이미지 읽기
    image = cv2.imread(input_image_path)
    height, width, _ = image.shape

    # 랜드마크 좌표를 픽셀 단위로 변환
    if "landmarks" in keypoints:
        landmarks = []
        for landmark in keypoints["landmarks"]:
            x = int(landmark["x"] * width)
            y = int(landmark["y"] * height)
            landmarks.append((x, y, landmark["visibility"]))

            # 가시성에 따라 키포인트 그림
            if landmark["visibility"] > 0.5:  # 임계값 조정 가능
                cv2.circle(image, (x, y), circle_radius, circle_color, -1)

        # 랜드마크 연결선 그리기
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]

                # 가시성 확인 후 선 그리기
                if start_point[2] > 0.5 and end_point[2] > 0.5:  # visibility 기준
                    cv2.line(image, (start_point[0], start_point[1]), 
                             (end_point[0], end_point[1]), line_color, line_thickness)

    # 결과 출력
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image with Keypoints and Connections")
    plt.show()

# 입력 이미지와 JSON 경로
input_image_path = "img_file/1.png"
input_json_path = "json_file/1.json"

# 함수 실행
draw_keypoints_and_connections(input_image_path, input_json_path)
