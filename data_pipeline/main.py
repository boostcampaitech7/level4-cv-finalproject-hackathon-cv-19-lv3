import os
import pandas as pd
import argparse
import glob
from copy import deepcopy
from pipeline import compare_video_pair, make_dataset
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge_path", 
        default="D:/naver_boostcamp/project/level4-cv-finalproject-hackathon-cv-19-lv3/yt_videos/Maratanghuru",
        help="챌린지에 대한 right영상 1개와 여러 wrong 영상이 위치한 폴더 경로"
    )

    parser.add_argument(
        "--system_prompt_path",
        default="D:/naver_boostcamp/project/level4-cv-finalproject-hackathon-cv-19-lv3/prompting/system_prompt.txt",
        help="system prompt로 사용할 지시문이 담겨있는 txt파일의 경로"
    )
    
    parser.add_argument(
        "--output_csv_path",
        default="./output.csv",
        help="output csv를 저장할 위치"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    challenge_path = args.challenge_path
    system_prompt_path = args.system_prompt_path
    output_csv_path = args.output_csv_path

    # system prompt 가져오기
    if system_prompt_path:
        with open(system_prompt_path, 'r', encoding='UTF8') as f:
            system_prompt = f.readlines()
        system_prompt = ''.join(system_prompt)

    # 지정 폴더에서 right video, wrong videos 경로 가져오기
    mp4_file_paths = glob.glob(os.path.join(challenge_path, '*.mp4'))
    indices = [index for index, string in enumerate(mp4_file_paths) if os.path.basename(string).startswith('right')]
    if not indices:
        raise OSError("There is no right video in the folder!")
    right_mp4_path = mp4_file_paths.pop(indices[0])
    wrong_mp4_list = mp4_file_paths

    total_result = pd.DataFrame({
        "System_Prompt": [], # 지시문
        "C_ID": [], #Conversation ID
        "T_ID": [], # Turn ID
        "Text": [], # 사용자가 말할 것으로 기대되는 모든 발화 내용
        "Completion": [] # CLOVA Studio가 답해야할 것으로 기대되는 모든 발화 내용
    })
    # wrong video와 right video를 하나씩 비교하며 dataframe 완성하기
    for wrong_video_path in tqdm(wrong_mp4_list):
        matched_dict_list = compare_video_pair(right_mp4_path, wrong_video_path, frame_interval=0.5)
        df = make_dataset(matched_dict_list, system_prompt, len(total_result))
        total_result = pd.concat([total_result, df], axis=0, ignore_index=True)
    
    total_result.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()