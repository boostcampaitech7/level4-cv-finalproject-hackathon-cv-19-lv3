import cv2
import multiprocessing

def capture_camera(camera_queue, camera_width, camera_height):
    """카메라 프레임을 읽어 큐에 전달"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        camera_queue.put(frame)

    camera.release()

def read_overlay(overlay_queue, overlay_path):
    """오버레이 동영상을 읽어 큐에 전달"""
    overlay_video = cv2.VideoCapture(overlay_path)
    overlay_width = int(overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    overlay_height = int(overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_fps = overlay_video.get(cv2.CAP_PROP_FPS)
    if overlay_fps == 0:  # FPS가 0인 경우 기본값 설정
        overlay_fps = 30

    frame_delay = int(1000 / overlay_fps)
    while True:
        ret, frame = overlay_video.read()
        if not ret:
            # 동영상 끝나면 다시 처음부터 재생
            overlay_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        overlay_queue.put((frame, frame_delay))

    overlay_video.release()

def main():
    # 오버레이 동영상 정보
    overlay_path = 'hong.mp4'
    temp_overlay_video = cv2.VideoCapture(overlay_path)
    overlay_width = int(temp_overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    overlay_height = int(temp_overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_overlay_video.release()

    # 큐 생성
    camera_queue = multiprocessing.Queue(maxsize=5)
    overlay_queue = multiprocessing.Queue(maxsize=5)

    # 프로세스 생성
    camera_process = multiprocessing.Process(target=capture_camera, args=(camera_queue, overlay_width, overlay_height))
    overlay_process = multiprocessing.Process(target=read_overlay, args=(overlay_queue, overlay_path))

    # 프로세스 시작
    camera_process.start()
    overlay_process.start()

    while True:
        if not camera_queue.empty() and not overlay_queue.empty():
            background_frame = camera_queue.get()
            overlay_frame, frame_delay = overlay_queue.get()

            # 합성
            background_frame_resized = cv2.resize(background_frame, (overlay_width, overlay_height))
            blended = cv2.addWeighted(background_frame_resized, 0.5, overlay_frame, 0.5, 0)

            cv2.imshow('Overlay', blended)

            # FPS에 맞게 대기
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

    # 프로세스 종료
    camera_process.terminate()
    overlay_process.terminate()
    camera_process.join()
    overlay_process.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
