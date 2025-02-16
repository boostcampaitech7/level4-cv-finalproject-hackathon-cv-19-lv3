import ffmpeg

def add_fade_effects1(input_file, output_file, fade_duration=2): # 원본 길이를 유지 (앞뒤를 fade로 변경)
    # Get video duration
    probe = ffmpeg.probe(input_file)
    duration = float(probe['format']['duration'])
    
    # Fade-in and fade-out filter parameters
    fade_in_filter = f'fade=t=in:st=0:d={fade_duration}'
    fade_out_filter = f'fade=t=out:st={duration - fade_duration}:d={fade_duration}'
    
    # Apply fade effects
    (
        ffmpeg
        .input(input_file)
        .filter('fade', t='in', st=0, d=fade_duration)
        .filter('fade', t='out', st=duration - fade_duration, d=fade_duration)
        .output(output_file)
        .run(overwrite_output=True)
    )
    
    print(f"Fade-in and fade-out applied successfully to {output_file}")

def add_fade_effects2(input_file, output_file, fade_duration=2): # 원본 영상을 유지 (앞뒤로 fade를 추가)
    # Get video duration
    probe = ffmpeg.probe(input_file)
    duration = float(probe['format']['duration'])
    
    # 원본 해상도 가져오기
    width = int(probe['streams'][0]['width'])
    height = int(probe['streams'][0]['height'])
    framerate = probe['streams'][0]['r_frame_rate']
    
    # 검은 화면 생성 (원본 해상도 유지)
    black_frame = ffmpeg.input(f'color=c=black:s={width}x{height}:r={framerate}', f='lavfi', t=fade_duration)
    
    # 페이드인, 페이드아웃 효과 추가
    faded_video = (
        ffmpeg
        .input(input_file)
        .filter('fade', t='in', st=0, d=fade_duration)
        .filter('fade', t='out', st=duration - fade_duration, d=fade_duration)
    )
    
    # 영상과 오디오 없이 연결 (오디오 스트림 제외)
    final_video = (
        ffmpeg.concat(black_frame, faded_video, black_frame, v=1, a=0)
        .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
        .run(overwrite_output=True)
    )
    
    print(f"Fade-in and fade-out applied successfully to {output_file}")

# Example usage
input_video = "dd.mp4"
output_video = "d.mp4"
add_fade_effects1(input_video, output_video, fade_duration=2)
