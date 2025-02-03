import cv2
import mediapipe as mp
import json

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints_to_json(input_file, output_json_path):
    # 이미지 읽기
    input_img = cv2.imread(input_file)
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Pose 추출
    results = pose.process(input_img_rgb)

    if results.pose_landmarks:
        # 키포인트 저장용 딕셔너리 초기화
        keypoints = {
            "landmarks": []
        }
        # 키포인트 데이터를 landmarks 리스트에 저장
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints["landmarks"].append({
                "id": id,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        
        # JSON 파일로 저장
        with open(output_json_path, "w") as json_file:
            json.dump(keypoints, json_file, indent=4)
        
        print(f"Keypoints saved to {output_json_path}")
    else:
        print("No pose landmarks detected.")

# 이미지 파일 경로
input_image_path = "3blue.jpg"
# 저장할 JSON 파일 경로
output_json_path = "3blue.json"

# 함수 실행
extract_keypoints_to_json(input_image_path, output_json_path)