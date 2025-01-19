import cv2
import time
import numpy as np
import json


with open("work_time_data.json", "r") as json_file:
    data = json.load(json_file)
average_work_time = data['average_work_time'] * 1000 # 한번 while문 돌 때 생기는 delay(ms)
original_time = data['total_duration']
elapsed_time = data['elapsed_time']
fps_amp = elapsed_time / original_time


# 배경: 카메라 영상
camera = cv2.VideoCapture(0)

# 오버랩할 동영상 불러오기
overlay_video = cv2.VideoCapture('hong.mp4')

# 동영상 크기 가져오기
overlay_width = int(overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
overlay_height = int(overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))


# 동영상 FPS 가져오기
overlay_fps = overlay_video.get(cv2.CAP_PROP_FPS) * fps_amp # delay를 고려
if overlay_fps == 0:  # FPS가 0인 경우 기본값 설정
    overlay_fps = 30

frame_delay = round(1000 / overlay_fps) # 한 프레임당 시간



# 카메라 해상도를 동영상 크기에 맞춤
camera.set(cv2.CAP_PROP_FRAME_WIDTH, overlay_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, overlay_height)
camera.set(cv2.CAP_PROP_FPS, overlay_fps)


# 시작 시간 기록
start_time = time.time()
camera_time_data = []
while True:
    s = time.time()
    ret, background_frame = camera.read()
    ret2, overlay_frame = overlay_video.read()
    e = time.time()
    camera_time_data.append(e-s)

    if not ret:
        break

    if not ret2:
        # 종료 시간 기록
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"작업 수행 시간(time.time으로 측정): {elapsed_time:.4f}초")
        start_time = time.time()
        print(f"평균 카메라 동작 타임 : {np.mean(camera_time_data)}")

        # 동영상이 끝났으면 다시 처음부터 재생
        overlay_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    

    # 카메라 프레임 크기를 동영상 크기에 맞춤
    background_frame_resized = cv2.resize(background_frame, (overlay_width, overlay_height))

    # 채널 확인 및 조정
    if background_frame_resized.shape[2] != overlay_frame.shape[2]:
        overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)

    # 합성 (투명도 적용)
    blended = cv2.addWeighted(background_frame_resized, 0.5, overlay_frame, 0.5, 0)

    cv2.imshow('Overlay', blended)

    # FPS에 맞게 대기 시간 설정
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

camera.release()
overlay_video.release()
cv2.destroyAllWindows()
