import os
import argparse
from pose_compare import get_detector, make_pose_jsons
from pose_feedback import json_to_prompt
from util import find_image_files



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="./images", help="image가 들어있는 폴더 경로")
    parser.add_argument("--model_size", type=int, default=2, help="mediapipe 모델 사이즈 0 ~ 2")
    parser.add_argument("--json_path", type=str, default='./results', help="landmarks를 json으로 저장할 위치")
    parser.add_argument("--prompt_path", type=str, default="./prompts", help="생성된 prompt를 저장할 위치")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    image_folder = args.img_path
    model_size = args.model_size
    json_result_folder = args.json_path
    prompt_path = args.prompt_path


    # get detector and make jsons
    detector = get_detector(model_size=model_size)
    image_files = find_image_files(image_folder)
    make_pose_jsons(image_files, detector=detector, result_folder=json_result_folder)


    # 같은 idx의 json에 대해 다음을 수행한다
    # target - right 비교 -> prompt생성
    # target - wrong 비교 -> prompt생성
    idx = 1
    folder_names = [os.path.splitext(os.path.basename(p))[0] for p in image_files]

    while True:
        target_pose_path = os.path.join(json_result_folder, f"target_pose_{idx}", "result.json")
        right_pose_path = os.path.join(json_result_folder, f"right_pose_{idx}", "result.json")
        wrong_pose_path = os.path.join(json_result_folder, f"wrong_pose_{idx}", "result.json")

        try:
            json_to_prompt(target_pose_path, right_pose_path, result_folder=prompt_path)
            json_to_prompt(target_pose_path, wrong_pose_path, result_folder=prompt_path)
        except:
            # 해당 idx 이상의 결과값이 없는 것으로 판단하고 종료
            break
        idx += 1

if __name__ == "__main__":
    main()