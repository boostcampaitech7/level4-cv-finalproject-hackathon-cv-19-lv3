import os
from copy import copy
import platform
import subprocess
import shutil
import numpy as np
from keypoint_map import KEYPOINT_MAPPING
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import random
import cv2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def download_model(model_size):
    if model_size == 0:
        file_name = "pose_landmarker_lite.task"
    elif model_size == 1:
        file_name = "pose_landmarker_full.task"
    else:
        file_name = "pose_landmarker_heavy.task"

    current_path = os.getcwd()
    model_folder = os.path.abspath(os.path.join(current_path, 'models'))
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    model_path = os.path.abspath(os.path.join(model_folder, file_name))

    # 모델 파일 이름 지정
    temp_model_file = copy(file_name)
    
    # 모델 경로에 파일이 있는지 확인
    if not os.path.exists(model_path):
        print(f"{model_path} 파일이 존재하지 않습니다. 다운로드를 시작합니다.")
        
        # 운영 체제 확인
        system = platform.system()
        
        if system == "Linux":
            command = (
                f"wget -O {temp_model_file} -q "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            )
        elif system == "Windows":
            command = (
                f"curl -o {temp_model_file} -s "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
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


def landmarks_to_dict(all_landmarks):
    landmark_dict = {}
    
    for i, landmarks in enumerate(all_landmarks):
        d = {j: {
                "name": KEYPOINT_MAPPING[j],
                "x": landmarks[j][0],
                "y": landmarks[j][1],
                "z": landmarks[j][2],
                "visibility": landmarks[j][3]
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