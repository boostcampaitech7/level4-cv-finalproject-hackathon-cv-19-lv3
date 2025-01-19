import json
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = ["image/1.png"]
BG_COLOR = (192, 192, 192)  # gray

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        
        if image is None:
            print(f"Failed to load image at {file}")
            continue  # 이미지를 읽을 수 없으면 건너뜁니다.
        
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Get pose landmarks and all detected points
dic = {}
for mark, data_point in zip(mp_pose.PoseLandmark, results.pose_landmarks.landmark):
    dic[mark.value] = dict(
        landmark=mark.name,
        x=data_point.x,
        y=data_point.y,
        z=data_point.z,
        visibility=data_point.visibility)

# Transform the dictionary into a JSON object
json_object = json.dumps(dic, indent=2)
