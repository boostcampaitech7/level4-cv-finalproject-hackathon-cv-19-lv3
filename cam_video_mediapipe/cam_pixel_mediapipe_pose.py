import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# 동영상 경로
input_video_path = "chal2.mp4"  # 입력 동영상 파일 경로
# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 입력 동영상의 FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이

# Mediapipe PoseLandmarker 설정
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

# 세그멘테이션 마스크를 저장할 리스트
segmentation_masks = []

# 동영상에서 세그멘테이션 마스크 추출
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 이미지를 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe Image로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # 세그멘테이션 마스크 생성
    result = detector.detect(mp_image)
    if result.segmentation_masks:
        # 첫 번째 마스크 선택
        segmentation_mask_image = result.segmentation_masks[0]
        mask_array = np.array(segmentation_mask_image.numpy_view())  # NumPy 배열로 변환
        mask = (mask_array * 255).astype(np.uint8)  # 0~255 범위로 변환

        # 마스크를 저장
        segmentation_masks.append(mask)

# 웹캠 영상 실행
cap.release()  # 동영상 캡처 종료

webcam_cap = cv2.VideoCapture(0)  # 웹캠 열기
i = 0
length = len(segmentation_masks)

# 웹캠 FPS를 얻어서 동기화
webcam_fps = webcam_cap.get(cv2.CAP_PROP_FPS)

# 웹캠이 열린 경우
while True:
    ret, frame = webcam_cap.read()
    if not ret:
        break

    # 웹캠 프레임 크기
    webcam_frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # 세그멘테이션 마스크 순차적으로 오버레이
    mask = segmentation_masks[i]

    # 마스크를 RGB로 변환
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 합성 (투명도 적용)
    blended = cv2.addWeighted(webcam_frame_resized, 0.9, mask_rgb, 0.1, 0)

    # 합성된 영상 화면에 표시
    cv2.imshow('Segmentation Overlay', blended)

    # 마스크 순차적으로 변경
    i = (i + 1) % length  # 이 부분을 수정하여 마스크가 계속 순차적으로 변경되도록 함

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_cap.release()
cv2.destroyAllWindows()
