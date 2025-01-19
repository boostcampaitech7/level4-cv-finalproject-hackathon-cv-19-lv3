import cv2
import numpy as np
from moviepy import VideoFileClip
import pygame
import time

class OverlapMovie:
    def __init__(self, video_path):
        self.camera = cv2.VideoCapture(0)
        self.overlay_video = VideoFileClip(video_path)


        ###### 비디오 - 카메라 동기화 프로세스
        # 동영상 크기 가져오기
        video_width, video_height = self.overlay_video.size

        # 카메라 해상도를 동영상 크기에 맞춤
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)


    def play_video_with_audio(self):
        # VideoFileClip을 사용하여 비디오 클립과 오디오를 로드
        audio = self.overlay_video.audio
        fps = self.overlay_video.fps  # 비디오의 FPS
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        # Pygame을 사용하여 오디오 재생
        pygame.mixer.init(frequency=int(audio.fps))
        audio.write_audiofile("temp_audio.wav", codec='pcm_s16le')  # 임시로 오디오 파일 저장
        pygame.mixer.music.load("temp_audio.wav")

        # 비디오 프레임 읽기
        frame_duration = 1 / fps  # 각 프레임의 지속 시간
        previous_time = time.time()  # 현재 시간 (초)

        # OpenCV를 사용하여 비디오 프레임을 읽고 출력
        for frame in self.overlay_video.iter_frames(fps=fps, dtype="uint8"):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB -> BGR로 변환 (OpenCV는 BGR 사용)

            ret, rec_frame = self.camera.read()
            if not ret:
                raise OSError("Camera is not working!!!")

            # 카메라 프레임을 비디오 크기에 맞게 리사이즈
            rec_frame_resized = cv2.resize(rec_frame, (frame.shape[1], frame.shape[0]))

            # 비디오 프레임과 웹캠 프레임을 합성 (오버레이)
            combined_frame = cv2.addWeighted(frame, 0.7, rec_frame_resized, 0.3, 0)

            # 첫 번째 프레임이 표시되었을 때 오디오 시작
            if pygame.mixer.music.get_busy() == 0:  # 오디오가 아직 시작되지 않았다면
                pygame.mixer.music.play()

            # 프레임 출력
            cv2.imshow('Video with Audio and Webcam', combined_frame)

            # 프레임 시간에 맞춰 대기
            current_time = time.time()
            elapsed_time = current_time - previous_time

            # 비디오 프레임 출력 후 대기 시간 계산 (오디오와 동기화)
            wait_time = max(5, int((frame_duration - elapsed_time) * 1000))  # 밀리초 단위
            previous_time = current_time

            # 'q' 키를 눌러 종료
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        # 종료 후 정리
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    video_path = '마라탕후루2.mp4'  # 비디오 파일 경로
    _movie = OverlapMovie(video_path)
    _movie.play_video_with_audio()
