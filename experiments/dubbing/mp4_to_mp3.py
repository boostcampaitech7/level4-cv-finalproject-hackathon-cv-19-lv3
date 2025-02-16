import subprocess

def extract_audio(input_file, output_file):
    try:
        command = [
            'ffmpeg',
            '-i', input_file,  # 입력 MP4 파일
            '-q:a', '0',       # 오디오 품질 (0 = 최고 품질)
            '-map', 'a',       # 오디오 스트림만 추출
            output_file        # 출력 MP3 파일
        ]
        subprocess.run(command, check=True)
        print(f"오디오 추출 완료: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")

# 사용 예제
# extract_audio('input.mp4', 'output.mp3')
# 사용 예제
input_video = 'luther.mp4'      # 입력 비디오 파일 경로
output_audio = 'luther.mp3'    # 추출할 MP3 파일 경로

extract_audio(input_video, output_audio)