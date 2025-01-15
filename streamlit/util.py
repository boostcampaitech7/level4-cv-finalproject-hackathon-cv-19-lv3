import numpy as np


# 간격 추가하여 두 프레임 이어붙이기
def concat_frames_with_spacing(frames, spacing=20, color=(0, 0, 0)):
    # 프레임 높이와 동일한 간격 이미지를 생성
    spacer = np.full((frames[0].shape[0], spacing, 3), color, dtype=np.uint8)
    
    # 프레임 + 간격 + 프레임 이어붙이기
    final_frames = []
    for frame in frames[:-1]:
        final_frames.append(frame)
        final_frames.append(spacer)
    final_frames.append(frames[-1])
    combined_frame = np.hstack(final_frames)

    return combined_frame