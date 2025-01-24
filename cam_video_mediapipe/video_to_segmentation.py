import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 동영상 경로 설정
input_video_path = "video_chal.mp4"  # 입력 동영상 파일 경로
output_video_path = "video_chal_seg.mp4"  # 출력 동영상 파일 경로

# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 입력 동영상의 FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이

# VideoWriter 초기화 (흑백 저장)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 코덱
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Mediapipe PoseLandmarker 설정
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

# 동영상 처리 루프
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
        # Mediapipe Image 객체를 NumPy 배열로 변환
        segmentation_mask_image = result.segmentation_masks[0]  # 첫 번째 마스크 선택
        mask_array = np.array(segmentation_mask_image.numpy_view())  # NumPy 배열로 변환
        mask = (mask_array * 255).astype(np.uint8)  # 0~255 범위로 변환

        # 마스크 동영상에 저장
        out.write(mask)

        # 실시간으로 화면에 표시 (옵션)
        # cv2.imshow('Segmentation Mask', mask)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# 리소스 정리
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Segmentation mask output saved to {output_video_path}")
