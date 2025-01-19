import cv2
import time
import numpy as np
import json


camera = cv2.VideoCapture(0)
overlay_video = cv2.VideoCapture('hong.mp4')

# 동영상 크기 가져오기
overlay_width = int(overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
overlay_height = int(overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))


# 동영상 FPS 가져오기
overlay_fps = overlay_video.get(cv2.CAP_PROP_FPS)
if overlay_fps == 0:  # FPS가 0인 경우 기본값 설정
    overlay_fps = 30
frame_delay = int(1000 / overlay_fps)  # FPS 기반 대기 시간 계산

# 동영상 정보 가져오기
frame_count = overlay_video.get(cv2.CAP_PROP_FRAME_COUNT)
total_duration = frame_count / overlay_fps


# 카메라 해상도를 동영상 크기에 맞춤
camera.set(cv2.CAP_PROP_FRAME_WIDTH, overlay_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, overlay_height)
camera.set(cv2.CAP_PROP_FPS, overlay_fps)



# 시작 시간 기록
start_time = time.time()
work_time_data = []
while True:
    s = time.time()
    ret, background_frame = camera.read()
    ret2, overlay_frame = overlay_video.read()

    if not ret:
        break

    if not ret2:
        # 종료 시간 기록
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_work_time = np.mean(work_time_data)

        # JSON 데이터 생성
        data = {
            "total_duration": round(total_duration, 4),
            "elapsed_time": round(elapsed_time, 4),  # 소수점 4자리까지 기록
            "average_work_time": round(avg_work_time, 4)  # 소수점 4자리까지 기록
        }

        # JSON 파일로 저장
        with open("work_time_data.json", "w") as json_file:
            json.dump(data, json_file, indent=4)  # JSON 파일에 기록 (들여쓰기 포함)
        print(f"작업 수행 시간(time.time으로 측정): {elapsed_time:.4f}초")
        print(f"평균 카메라 동작 타임 : {avg_work_time}")
        break
    

    # 카메라 프레임 크기를 동영상 크기에 맞춤
    background_frame_resized = cv2.resize(background_frame, (overlay_width, overlay_height))

    # 채널 확인 및 조정
    if background_frame_resized.shape[2] != overlay_frame.shape[2]:
        overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)

    # 합성 (투명도 적용)
    blended = cv2.addWeighted(background_frame_resized, 0.5, overlay_frame, 0.5, 0)

    cv2.imshow('Overlay', blended)


    e = time.time()
    work_time_data.append(e-s)
    # FPS에 맞게 대기 시간 설정
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

camera.release()
overlay_video.release()
cv2.destroyAllWindows()