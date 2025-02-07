import sys
import random
from tqdm import tqdm
sys.path.append("./")
import pandas as pd
import numpy as np
from scipy.stats import norm

from dance_scoring.detector import PoseDetector, post_process_pose_landmarks
from dance_scoring.similarity_with_frames import *
from feedback.pose_compare import extract_pose_landmarks
from feedback.pose_feedback import json_to_prompt, generate_feedback, generate_korean_feedback


english_to_korean = {
    "head": "머리",
    "shoulder": "어깨",
    "left_arm": "왼쪽 팔",
    "right_arm": "오른쪽 팔",
    "left_elbow": "왼쪽 팔목",
    "right_elbow": "오른쪽 팔목",
    "left_leg": "왼쪽 다리",
    "right_leg": "오른쪽 다리",
    "left_knee": "왼쪽 무릎",
    "right_knee": "오른쪽 무릎"
}

def compare_video_pair(right_video_path, wrong_video_path, frame_interval=0.5):
    estimate_class = PoseDetector()
    right_original_video_frames, right_pose_landmarker_results, right_shape, right_fps = estimate_class.get_video_landmarks(right_video_path)

    wrong_original_video_frames, wrong_pose_landmarker_results, wrong_shape, wrong_fps = estimate_class.get_video_landmarks(wrong_video_path)
    
    # None값 처리
    right_pose_landmarker_results = post_process_pose_landmarks(right_pose_landmarker_results)
    wrong_pose_landmarker_results = post_process_pose_landmarks(wrong_pose_landmarker_results)

    # keypoints L2 정규화
    right_keypoints = get_normalized_keypoints(right_pose_landmarker_results, *right_shape)
    wrong_keypoints = get_normalized_keypoints(wrong_pose_landmarker_results, *wrong_shape)

    # 유사도 및 시각화 데이터 계산
    distance, average_cosine_similarity, average_euclidean_distance, average_oks, average_pck, pairs = calculate_similarity_with_visualization(
        right_keypoints, wrong_keypoints
    )
    
    # 매치된 pair끼리 frame, keypoint 저장
    matched_dict_list = []
    for idx1, frame in enumerate(right_original_video_frames):
        if idx1 % (right_fps * frame_interval) != 0:
            continue

        idx2 = get_center_pair_frames(pairs, idx1, matched_idx=0)
        matched_dict_list.append({
            'right_idx': idx1,
            'wrong_idx': idx2,
            'time': right_fps * idx1,
            'right_frame': frame,
            'wrong_frame': wrong_original_video_frames[idx2],
            'right_keypoint': right_pose_landmarker_results[idx1],
            'wrong_keypoint': wrong_pose_landmarker_results[idx2],
            'right_shape': right_shape,
            'wrong_shape': wrong_shape,
            'scores': {
                'average_cosine_similarity': average_cosine_similarity,
                'average_euclidean_distance': average_euclidean_distance,
                'average_oks': average_oks,
                'average_pck': average_pck
            }
        })
    
    return matched_dict_list

def delete_low_difference(result_dict, threshold):
    low_diff_keys = []
    for k, v in result_dict.items():
        if abs(v) < threshold:
            low_diff_keys.append(k)
    
    for k in low_diff_keys:
        del result_dict[k]


def get_feedback_from_keypoints(match_info_dict):
    # dictionary로부터 필요한 정보 가져오기
    right_keypoint, right_shape = match_info_dict['right_keypoint'], match_info_dict['right_shape']
    wrong_keypoint, wrong_shape = match_info_dict['wrong_keypoint'], match_info_dict['wrong_shape']

    # 사전정의된 알고리즘에 따라 관절 각도 정보를 dictionary로 가져옴
    right_pose_json = extract_pose_landmarks(right_keypoint, right_shape[1], right_shape[0])
    wrong_pose_json = extract_pose_landmarks(wrong_keypoint, wrong_shape[1], wrong_shape[0])

    # 각도 정보를 비교하여 수치적인 차이와 그에 해당하는 자연어 피드백을 dictionary형태로 가져옴
    differences = json_to_prompt(right_pose_json, wrong_pose_json)
    return differences


def numeric_to_text(numeric_result_json):
    for k, v in numeric_result_json.items():
        if v == 0:
            numeric_result_json[k] = "목표 자세와 차이가 없습니다."
        else:
            numeric_result_json[k] = f"목표 자세에 대해 {v}만큼의 각도 차이가 있습니다."
    return numeric_result_json


def input_prompt_from_dict(difference_dict):
    '''
    difference_dict : {head_difference: 50, ...}과 같은 형태의 dictionary
    '''
    input_sentence = ''
    for k, v in difference_dict.items():
        input_sentence += f"{' '.join(k.split('_'))}: {v} "
    return input_sentence


def output_sentence_from_dict(feedback_dict):
    startings = [
        "동작 차이를 기반으로 피드백을 드리도록 하겠습니다.",
        "동작 분석을 기반으로 피드백을 제공해 드리겠습니다.",
        "개선점을 안내해 드릴게요.",
        "정확한 동작 분석을 통해 개선할 점을 알려드리겠습니다.",
        "댄스 동작을 비교하여 최적의 피드백을 드릴게요.",
        "자세 차이를 바탕으로 더 나은 동작을 위한 피드백을 드리겠습니다.",
        "댄스 퍼포먼스를 향상시킬 수 있도록 피드백을 시작하겠습니다."
    ]

    endings = [
        "나머지 동작는 좋아요! 계속해서 발전해봅시다!",
        "좋은 동작입니다! 앞으로도 꾸준히 연습해볼까요?",
        "지금도 훌륭해요! 조금씩 더 다듬어 가봅시다!",
        "완벽에 가까워지고 있어요! 계속 노력해볼게요!",
        "점점 더 좋아지고 있어요! 계속 밀고 나가봅시다!",
        "좋은 흐름이에요! 이대로 쭉 가봅시다!",
        "자세가 많이 발전했어요! 다음 목표를 향해 가볼까요?",
        "멋진 동작이에요! 앞으로도 함께 최선을 다해봐요!"
    ]

    if "perfect_msg" in feedback_dict:
        output_sentence = feedback_dict["perfect_msg"]
    else:
        # output_sentence = f"{random.choice(startings) }"
        output_sentence = ""
        for k, v in feedback_dict.items():
            output_sentence += f"{v} "
        # output_sentence += random.choice(endings)
    return output_sentence


def make_dataset(matched_dict_list, system_prompt, start_CID=0, threshold=30, ignore_low_difference=True, do_numeric_to_text=False):
    df = {
        "System_Prompt": [], # 지시문
        "C_ID": [], #Conversation ID
        "T_ID": [], # Turn ID
        "Text": [], # 사용자가 말할 것으로 기대되는 모든 발화 내용
        "Completion": [] # CLOVA Studio가 답해야할 것으로 기대되는 모든 발화 내용
    }

    for idx, matched_dict in enumerate(matched_dict_list):
        differences = get_feedback_from_keypoints(matched_dict)
        feedbacks = generate_korean_feedback(differences, threshold=threshold)

        # 낮은 값들 거르는지 여부 보고 input에서 제외
        if ignore_low_difference:
            delete_low_difference(differences, threshold)

        # input prompt를 json으로부터 작성
        input_sentence = input_prompt_from_dict(numeric_to_text(differences))

        # output sentence를 dict로부터 작성
        output_sentence = output_sentence_from_dict(feedbacks)

        df["C_ID"].append(idx+start_CID)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)
        df["Text"].append(input_sentence)
        df["Completion"].append(output_sentence)
    
    df = pd.DataFrame(df)
    return df


# 표준편차를 계산하는 함수
def calculate_std_dev(threshold):
    """ threshold 값 내에 들어올 확률이 60%가 되도록 std_dev를 조정 """
    return threshold / norm.ppf(0.8)  # 0.8는 (0.6 + 0.5)


# 랜덤 값 생성 함수
def generate_random_value(mean, min_val, max_val, threshold):
    std_dev = calculate_std_dev(threshold)
    while True:
        value = int(np.random.normal(mean, std_dev))
        if min_val <= value <= max_val:
            return value


def make_random_dataset(total_data_cnt, system_prompt, max_threshold=30, perfect_rate=0.1, ignore_low_difference=True, do_numeric_to_text=False):
    df = {
        "System_Prompt": [], # 지시문
        "C_ID": [], #Conversation ID
        "T_ID": [], # Turn ID
        "Text": [], # 사용자가 말할 것으로 기대되는 모든 발화 내용
        "Completion": [] # CLOVA Studio가 답해야할 것으로 기대되는 모든 발화 내용
    }

    # 범위 정의
    ranges = {
        "head_difference": (-70, 70),
        "shoulder_difference": (-100, 100),
        "left_arm_angle_difference": (-140, 140),
        "right_arm_angle_difference": (-140, 140),
        "left_elbow_angle_difference": (-140, 140),
        "right_elbow_angle_difference": (-140, 140),
        "left_leg_angle_difference": (-90, 90),
        "right_leg_angle_difference": (-90, 90),
        "left_knee_angle_difference": (-140, 140),
        "right_knee_angle_difference": (-140, 140),
    }

    min_threshold = 10
    for idx in tqdm(range(total_data_cnt)):
        # 랜덤 값 생성
        threshold = random.randint(min_threshold, max_threshold)
        differences = {key: generate_random_value(0, *ranges[key], threshold) for key in ranges}
        if np.random.rand() < perfect_rate:
            for k, v in differences.items():
                differences[k] = int(np.random.uniform(-1, 1) * threshold)
        differences['threshold'] = threshold

        feedbacks = generate_korean_feedback(differences, threshold=threshold)

        # 낮은 값들 거르는지 여부 보고 input에서 제외
        if ignore_low_difference:
            delete_low_difference(differences, threshold)

        # input prompt를 dict으로부터 작성
        if do_numeric_to_text:
            input_sentence = str(numeric_to_text(differences))
        else:
            input_sentence = str(differences)

        # output sentence를 dict로부터 작성
        output_sentence = output_sentence_from_dict(feedbacks)

        df["C_ID"].append(idx)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)
        df["Text"].append(input_sentence)
        df["Completion"].append(output_sentence)
    

    df = pd.DataFrame(df)
    return df