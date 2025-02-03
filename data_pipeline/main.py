import os
import pandas as pd
import argparse
import glob
from pipeline import compare_video_pair, make_dataset, make_random_dataset
from tqdm import tqdm

def str_to_bool(value):
    if value.lower() in ("True", "true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("False", "false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge_path",
        type=str, 
        default="",
        help="챌린지에 대한 right영상 1개와 여러 wrong 영상이 위치한 폴더 경로. 빈 문자열일 경우 랜덤 생성이 수행됨."
    )

    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default="./prompts/structured_system_prompt_short.txt",
        help="system prompt로 사용할 지시문이 담겨있는 txt파일의 경로"
    )
    
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="./clova_datasets",
        help="output csv를 저장할 위치"
    )

    parser.add_argument(
        "--instruction_dataset",
        type=str_to_bool,
        default=False,
        help="데이터셋 구성을 instruction dataset으로 할 것인지 아닌지 여부(차이점은 clova docs 참고)"
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=30,
        help='얼마만큼의 각도차이가 나야 피드백 대상으로 선정되는지를 정하는 threshold값'
    )

    parser.add_argument(
        '--ignore_low_difference',
        type=str_to_bool,
        default=True,
        help="threshold보다 낮은 difference의 정보를 input에 넣을지 여부."
    )

    parser.add_argument(
        '--do_numeric_to_text',
        type=str_to_bool,
        default=False,
        help="수치를 문장화시킬지 여부."
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    challenge_path = args.challenge_path
    system_prompt_path = args.system_prompt_path
    output_csv_path = args.output_csv_path
    instruction_dataset = args.instruction_dataset
    threshold = args.threshold
    ignore_low_difference = args.ignore_low_difference
    do_numeric_to_text= args.do_numeric_to_text

    # system prompt 가져오기
    if system_prompt_path:
        with open(system_prompt_path, 'r', encoding='UTF8') as f:
            system_prompt = f.readlines()
        system_prompt = ''.join(system_prompt)
    else:
        system_prompt = ''

    total_result = None

    # 지정 폴더에서 right video, wrong videos 경로 가져오기
    if challenge_path:
        mp4_file_paths = glob.glob(os.path.join(challenge_path, '*.mp4'))
        indices = [index for index, string in enumerate(mp4_file_paths) if os.path.basename(string).startswith('right')]
        if not indices:
            raise OSError("There is no right video in the folder!")
        right_mp4_path = mp4_file_paths.pop(indices[0])
        wrong_mp4_list = mp4_file_paths

        # wrong video와 right video를 하나씩 비교하며 dataframe 완성하기
        for wrong_video_path in tqdm(wrong_mp4_list):
            if isinstance(total_result, pd.DataFrame):
                data_idx = len(total_result)
            else:
                data_idx = 0

            matched_dict_list = compare_video_pair(right_mp4_path, wrong_video_path, frame_interval=0.5)
            df = make_dataset(matched_dict_list, system_prompt, data_idx, threshold=threshold, ignore_low_difference=ignore_low_difference, do_numeric_to_text=do_numeric_to_text)
            if isinstance(total_result, pd.DataFrame):
                total_result = pd.concat([total_result, df], axis=0, ignore_index=True)
            else:
                total_result = df

    # 랜덤하게 데이터 생성하는 파트
    else:
        print("challenge_path를 지정하지 않았기 때문에 랜덤 데이터로 데이터셋을 구성합니다.")
        random_cnt = input("총 몇개의 데이터를 생성할 것인지 입력해주세요(default 1000): ")
        try:
            random_cnt = int(random_cnt)
        except:
            random_cnt = 1000
        total_result = make_random_dataset(random_cnt, system_prompt, threshold=threshold, ignore_low_difference=ignore_low_difference, do_numeric_to_text=do_numeric_to_text)

    # instruction 형식이 아닌 경우 text, completion 열만 필요함
    if not instruction_dataset:
        total_result = total_result[['Text', 'Completion']]
    
    if not os.path.exists(output_csv_path):
        os.mkdir(output_csv_path)
    total_result.to_csv(os.path.join(output_csv_path, "output.csv"), index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    main()