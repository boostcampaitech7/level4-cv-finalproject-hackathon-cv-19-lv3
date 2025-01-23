import os
from copy import deepcopy
import platform
import subprocess
import shutil
import numpy as np
from keypoint_map import KEYPOINT_MAPPING, NORMALIZED_LANDMARK_KEYS
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import random
import cv2
from collections import namedtuple

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def download_model(model_size):
    if model_size == 0:
        file_name = "pose_landmarker_lite.task"
        download_link = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

    elif model_size == 1:
        file_name = "pose_landmarker_full.task"
        download_link = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    else:
        file_name = "pose_landmarker_heavy.task"
        download_link = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

    current_path = os.getcwd()
    model_folder = os.path.abspath(os.path.join(current_path, 'models'))
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    model_path = os.path.abspath(os.path.join(model_folder, file_name))

    # 모델 파일 이름 지정
    temp_model_file = deepcopy(file_name)
    
    # 모델 경로에 파일이 있는지 확인
    if not os.path.exists(model_path):
        print(f"{model_path} 파일이 존재하지 않습니다. 다운로드를 시작합니다.")
        
        # 운영 체제 확인
        system = platform.system()
        
        if system == "Linux":
            command = (
                f"wget -O {temp_model_file} -q {download_link}"
            )
        elif system == "Windows":
            command = (
                f"curl -o {temp_model_file} -s {download_link}"
            )
        else:
            raise OSError("지원하지 않는 운영 체제입니다.")
        
        # 명령어 실행
        try:
            subprocess.run(command, shell=True, check=True)
            print("모델 다운로드가 완료되었습니다.")
            
            # 다운로드한 파일을 model_path로 이동
            shutil.move(temp_model_file, model_path)
            print(f"{temp_model_file} 파일을 {model_path}로 이동하였습니다.")
        except subprocess.CalledProcessError as e:
            print(f"모델 다운로드 중 오류가 발생했습니다: {e}")
        except Exception as e:
            print(f"파일 이동 중 오류가 발생했습니다: {e}")
    else:
        print(f"{model_path} 파일이 이미 존재합니다.")
    return model_path

def get_max_height_from_frames(frames):
    max_frame_height = 0
    for frame in frames:
        max_frame_height = np.max([max_frame_height, frame.shape[0]])
    return max_frame_height


# 간격 추가하여 두 프레임 이어붙이기
def concat_frames_with_spacing(frames, max_frame_height=None, spacing=20, color=(0, 0, 0)):
    # 서로 height가 다른 경우를 위한 패딩
    if max_frame_height:
        for i, frame in enumerate(frames):
            total_pad_length = max_frame_height - frame.shape[0]
            if total_pad_length > 0:
                frames[i] = np.pad(frame, ([total_pad_length//2, total_pad_length//2 + total_pad_length%2], [0, 0]), mode='constant')
    
        # 프레임 높이와 동일한 간격 이미지를 생성
        spacer = np.full((max_frame_height, spacing, 3), color, dtype=np.uint8)
    else:
        spacer = np.full((frames[0].shape[0], spacing, 3), color, dtype=np.uint8)

    # 프레임 + 간격 + 프레임 이어붙이기
    final_frames = []
    for frame in frames[:-1]:
        final_frames.append(frame)
        final_frames.append(spacer)
    final_frames.append(frames[-1])
    combined_frame = np.hstack(final_frames)

    return combined_frame


def landmarks_to_dict(all_landmarks):
    landmark_dict = {}
    
    for i, landmarks in enumerate(all_landmarks):
        d = {j: {
                "name": KEYPOINT_MAPPING[j],
                "x": landmarks[j].x,
                "y": landmarks[j].y,
                "z": landmarks[j].z,
                "visibility": landmarks[j].visibility
            } for j in KEYPOINT_MAPPING.keys()}
        landmark_dict[i] = d
    return landmark_dict


def draw_landmarks_on_image(rgb_image, detection_result, landmarks_c=(234,63,247), connection_c=(117,249,77), 
                    thickness=10, circle_r=10):
  try:
      pose_landmarks_list = detection_result.pose_landmarks
  except:
      pose_landmarks_list = [detection_result]
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_utils.DrawingSpec(landmarks_c, thickness, circle_r),
      solutions.drawing_utils.DrawingSpec(connection_c, thickness, circle_r))
  return annotated_image



def image_alpha_control(image, alpha=0.5):
    background = np.zeros_like(image, dtype=np.uint8)  # 검정색 배경 생성
    blended_image = cv2.addWeighted(image, alpha, background, 1 - alpha, 0)
    return blended_image


def draw_circle_on_image(image: np.ndarray, normalized_x: float, normalized_y: float, r: int, color=(255, 0, 0), thickness=2):
    """
    Draws a circle on a given image at a position defined by normalized coordinates.

    Args:
        image (np.ndarray): The input image (H, W, C).
        normalized_x (float): Normalized x-coordinate (0 ~ 1).
        normalized_y (float): Normalized y-coordinate (0 ~ 1).
        r (int): Radius of the circle.
        color (tuple): Color of the circle in BGR format. Default is blue (255, 0, 0).
        thickness (int): Thickness of the circle outline. Use -1 for a filled circle.
    """
    # 이미지 크기 계산
    height, width = image.shape[:2]
    
    # 정규화된 좌표를 실제 픽셀 좌표로 변환
    x = int(normalized_x * width)
    y = int(normalized_y * height)
    
    # 원 그리기
    cv2.circle(image, (x, y), r, color, thickness)


def fill_None_from_landmarks(all_landmarks, fill_value=1.):
    NormalizedLandmark = namedtuple('NormalizedLandmark', NORMALIZED_LANDMARK_KEYS)
    none_fill_value = [NormalizedLandmark(**{k:.0 for k in NORMALIZED_LANDMARK_KEYS}) for _ in range(len(KEYPOINT_MAPPING))]
    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            all_landmarks[i] = none_fill_value
    return all_landmarks


def get_closest_frame(time_in_seconds, total_frames, fps):
    """
    주어진 시간과 FPS, 총 프레임 수를 기반으로 가장 가까운 프레임 번호를 계산합니다.
    
    Parameters:
    - time_in_seconds (int): 시간 (초 단위)
    - total_frames (int): 총 프레임 수
    - fps (int): 초당 프레임 수 (Frames Per Second)
    
    Returns:
    - closest_frame (int): 주어진 시간에 가장 가까운 프레임 번호
    """
    # 계산된 프레임 번호
    calculated_frame = time_in_seconds * fps
    
    # 프레임 번호가 총 프레임 범위를 초과하지 않도록 조정
    closest_frame = min(max(0, round(calculated_frame)), total_frames - 1)
    
    return closest_frame