import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 동영상 파일 설정
overlay_video = cv2.VideoCapture('video_chal.mp4')

# 동영상 크기 및 FPS 가져오기
overlay_width = int(overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
overlay_height = int(overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
overlay_fps = overlay_video.get(cv2.CAP_PROP_FPS)
if overlay_fps == 0:  # FPS 기본값 설정
    overlay_fps = 30
frame_delay = int(1000 / overlay_fps)

# 카메라 설정 및 크기 조정
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, overlay_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, overlay_height)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while camera.isOpened():
        ret, background_frame = camera.read()
        ret2, overlay_frame = overlay_video.read()

        if not ret:
            break

        if not ret2:
            # 동영상이 끝났으면 다시 처음부터 재생
            overlay_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, overlay_frame = overlay_video.read()

        # 좌우반전 복구
        background_frame = cv2.flip(background_frame, 1)

        # 카메라 프레임 크기를 동영상 크기에 맞춤
        background_frame_resized = cv2.resize(background_frame, (overlay_width, overlay_height))

        # 동영상과 카메라 프레임 합성
        blended = cv2.addWeighted(background_frame_resized, 0.5, overlay_frame, 0.5, 0)

        # MediaPipe를 통해 Pose 추출
        image_rgb = cv2.cvtColor(background_frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Pose 랜드마크가 있으면 화면에 표시
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                blended,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # 화면에 출력
        cv2.imshow('Overlay with Pose Estimation', blended)

        # FPS에 맞게 대기 시간 설정
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

camera.release()
overlay_video.release()
cv2.destroyAllWindows()
