import os
from copy import deepcopy
import platform
import subprocess
import shutil

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