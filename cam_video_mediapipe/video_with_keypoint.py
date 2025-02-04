import cv2
import mediapipe as mp

# MediaPipe Pose 모델 로드
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 동영상 파일 열기 ("video.mp4"를 원하는 동영상 파일 경로로 변경)
video_path = "video_no.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 비디오 재생 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오가 끝나면 종료
    
    # BGR을 RGB로 변환 (MediaPipe는 RGB 이미지를 사용)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Pose 모델을 사용하여 keypoints 감지
    results = pose.process(frame_rgb)
    
    # keypoints가 감지된 경우 화면에 표시
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    
    # 결과 영상 표시
    cv2.imshow("MediaPipe Pose Keypoints", frame)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 정리
cap.release()
cv2.destroyAllWindows()
