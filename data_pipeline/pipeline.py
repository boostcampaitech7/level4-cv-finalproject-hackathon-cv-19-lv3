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
from feedback.pose_feedback import json_to_prompt, generate_korean_feedback
from feedback import pose_feedback_final
import config


english_to_korean = {
    "head": "머리",
    "shoulder": "어깨",
    "left arm": "왼쪽 팔",
    "right arm": "오른쪽 팔",
    "left elbow": "왼쪽 팔목",
    "right elbow": "오른쪽 팔목",
    "left leg": "왼쪽 다리",
    "right leg": "오른쪽 다리",
    "left knee": "왼쪽 무릎",
    "right knee": "오른쪽 무릎"
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
        if k == 'threshold':
            numeric_result_json[k] = f"피드백을 주어야하는 임계치는 {v}입니다."
            continue

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
        "동작 차이를 기반으로 피드백을 드리도록 하겠습니다.\n\n",
        "동작 분석을 기반으로 피드백을 제공해 드리겠습니다.\n\n",
        "개선점을 안내해 드릴게요.\n\n",
        "정확한 동작 분석을 통해 개선할 점을 알려드리겠습니다.\n\n",
        "댄스 동작을 비교하여 최적의 피드백을 드릴게요.\n\n",
        "자세 차이를 바탕으로 더 나은 동작을 위한 피드백을 드리겠습니다.\n\n",
        "댄스 퍼포먼스를 향상시킬 수 있도록 피드백을 시작하겠습니다.\n\n"
    ]


    good_endings = [
        "\n나머지 동작는 좋아요! 계속해서 발전해봅시다!",
        "\n좋은 동작입니다! 앞으로도 꾸준히 연습해볼까요?",
        "\n완벽에 가까워지고 있어요! 계속 노력해볼게요!",
        "\n점점 더 좋아지고 있어요! 계속 밀고 나가봅시다!",
        "\n좋은 흐름이에요! 이대로 쭉 가봅시다!",
        "\n자세가 많이 발전했어요! 다음 목표를 향해 가볼까요?",
    ]

    bad_endings = [
        "\n자세가 아직은 많이 좋지 않네요 더 정진해봅시다!",
        "\n아직은 부족한 부분이 보이지만 꾸준히 연습하면 좋아질 거예요!",
        "\n자세를 조금 더 신경 쓰면 훨씬 좋아질 거예요! 계속 연습해봐요!",
        "\n연습을 조금 더 하면 동작이 훨씬 자연스러워질 거예요! 화이팅!",
        "\n동작의 연결이 자연스럽게 이어질 수 있도록 신경 써보면 좋을 것 같아요!"
    ]

    if "perfect msg" in feedback_dict:
        output_sentence = feedback_dict["perfect msg"]
    else:
        output_sentence = f"{random.choice(startings) }"
        feedback_sentences = []


        # difference 총합에 따라 ending을 결정
        if len(feedback_dict) >= 5:
            ending = random.choice(bad_endings)
        else:
            ending = random.choice(good_endings)

        # 머리는 개별 설명
        if 'head' in feedback_dict:
            feedback_sentences.append(feedback_dict['head'] + '\n')
        
        # 왼팔에 대한 부분 묶어서 설명
        if 'shoulder' in feedback_dict and '왼쪽' in feedback_dict['shoulder']:
            s = '다음으로는 ' if feedback_sentences else ''

            if 'left arm' in feedback_dict and 'left elbow' in feedback_dict:
                s += feedback_dict["shoulder"].replace('주세요.', "주시고, ")
                s += feedback_dict['left arm'].replace('을', '은').replace('주세요.', '주신 다음에 ')
                s += feedback_dict['left elbow'].replace('를', '도')
            elif 'left arm' in feedback_dict:
                s += feedback_dict["shoulder"].replace('주세요.', "주시고, ")
                s += feedback_dict['left arm'].replace('을', '은')
            elif 'left elbow' in feedback_dict:
                s += feedback_dict["shoulder"].replace('주세요.', "주시고, ")
                s += feedback_dict['left elbow'].replace('를', '는')
            else:
                s += feedback_dict["shoulder"]
            feedback_sentences.append(s + '\n')

        elif 'shoulder' not in feedback_dict:
            s = '다음으로는 ' if feedback_sentences else ''
            if 'left arm' in feedback_dict and 'left elbow' in feedback_dict:
                s += feedback_dict['left arm'].replace('주세요.', '주신 다음에 ')
                s += feedback_dict['left elbow'].replace('를', '도')
            elif 'left arm' in feedback_dict:
                s += feedback_dict['left arm']
            elif 'left elbow' in feedback_dict:
                s += feedback_dict['left elbow']
            else:
                s = ''
            
            if s:
                feedback_sentences.append(s + '\n')
        else:
            pass
        
        # 오른팔에 대한 부분 묶어서 설명
        if 'shoulder' in feedback_dict and '오른쪽' in feedback_dict['shoulder']:
            s = '계속해서 ' if feedback_sentences else ''

            if 'right arm' in feedback_dict and 'right elbow' in feedback_dict:
                s += feedback_dict["shoulder"].replace('주세요.', "주시고, ")
                s += feedback_dict['right arm'].replace('을', '은').replace('주세요.', '주신 다음에 ')
                s += feedback_dict['right elbow'].replace('를', '도')
            elif 'right arm' in feedback_dict:
                s += feedback_dict["shoulder"].replace('주세요.', "주시고, ")
                s += feedback_dict['right arm'].replace('을', '은')
            elif 'right elbow' in feedback_dict:
                s += feedback_dict["shoulder"].replace('주세요.', "주시고, ")
                s += feedback_dict['right elbow'].replace('를', '는')
            else:
                s += feedback_dict["shoulder"]
            feedback_sentences.append(s + '\n')

        elif 'shoulder' not in feedback_dict:
            s = '계속해서 ' if feedback_sentences else ''
            if 'right arm' in feedback_dict and 'right elbow' in feedback_dict:
                s += feedback_dict['right arm'].replace('주세요.', '주신 다음에 ')
                s += feedback_dict['right elbow'].replace('를', '도')
            elif 'right arm' in feedback_dict:
                s += feedback_dict['right arm']
            elif 'right elbow' in feedback_dict:
                s += feedback_dict['right elbow']
            else:
                s = ''
            
            if s:
                feedback_sentences.append(s + '\n')
        else:
            pass
        
        # 왼발에 대한 부분 묶어서 설명
        if 'left leg' in feedback_dict or 'left knee' in feedback_dict:
            s = '그리고 ' if feedback_sentences else ''
            if 'left leg' in feedback_dict and 'left knee' in feedback_dict:
                s += feedback_dict['left leg'].replace('주세요.', '주시고, ')
                s += feedback_dict['left knee'].replace('를', '는')
            elif 'left leg' in feedback_dict:
                s += feedback_dict['left leg']
            elif 'left knee' in feedback_dict:
                s += feedback_dict['left knee']

            feedback_sentences.append(s + '\n')

        # 오른발에 대한 부분 묶어서 설명
        if 'right leg' in feedback_dict or 'right knee' in feedback_dict:
            s = '마지막으로 ' if feedback_sentences else ''

            if 'right leg' in feedback_dict and 'right knee' in feedback_dict:
                s += feedback_dict['right leg'].replace('주세요.', '주시고, ')
                s += feedback_dict['right knee'].replace('를', '도')
            elif 'right leg' in feedback_dict:
                s += feedback_dict['right leg']
            elif 'right knee' in feedback_dict:
                s += feedback_dict['right knee']

            feedback_sentences.append(s + '\n')
        output_sentence += ''.join(feedback_sentences)
        output_sentence += ending

    return output_sentence

def output_sentence_from_dict_simple(feedback_dict):
    startings = [
        "동작 차이를 기반으로 피드백을 드리도록 하겠습니다.",
        "동작 분석을 기반으로 피드백을 제공해 드리겠습니다.",
        "개선점을 안내해 드릴게요.",
        "정확한 동작 분석을 통해 개선할 점을 알려드리겠습니다.",
        "댄스 동작을 비교하여 최적의 피드백을 드릴게요.",
        "자세 차이를 바탕으로 더 나은 동작을 위한 피드백을 드리겠습니다.",
        "댄스 퍼포먼스를 향상시킬 수 있도록 피드백을 시작하겠습니다."
    ]

    good_endings = [
        "나머지 동작는 좋아요! 계속해서 발전해봅시다!",
        "좋은 동작입니다! 앞으로도 꾸준히 연습해볼까요?",
        "완벽에 가까워지고 있어요! 계속 노력해볼게요!",
        "점점 더 좋아지고 있어요! 계속 밀고 나가봅시다!",
        "좋은 흐름이에요! 이대로 쭉 가봅시다!",
        "자세가 많이 발전했어요! 다음 목표를 향해 가볼까요?"
    ]

    bad_endings = [
        "자세가 아직은 많이 좋지 않네요 더 정진해봅시다!",
        "아직은 부족한 부분이 보이지만 꾸준히 연습하면 좋아질 거예요!",
        "자세를 조금 더 신경 쓰면 훨씬 좋아질 거예요! 계속 연습해봐요!",
        "연습을 조금 더 하면 동작이 훨씬 자연스러워질 거예요! 화이팅!",
        "동작의 연결이 자연스럽게 이어질 수 있도록 신경 써보면 좋을 것 같아요!"
    ]


    if len(feedback_dict) >= 5:
        ending = random.choice(bad_endings)
    else:
        ending = random.choice(good_endings)

    if "perfect msg" in feedback_dict:
        output_sentence = feedback_dict["perfect msg"]
    else:
        output_sentence = random.choice(startings) + ' '
        for k, v in feedback_dict.items():
            output_sentence += f"{v} "
        output_sentence += ending
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
        "head difference": (-70, 70),
        "shoulder difference": (-100, 100),
        "left arm angle difference": (-140, 140),
        "right arm angle difference": (-140, 140),
        "left elbow angle difference": (-140, 140),
        "right elbow angle difference": (-140, 140),
        "left leg angle difference": (-120, 120),
        "right leg angle difference": (-120, 120),
        "left knee angle difference": (-140, 140),
        "right knee angle difference": (-140, 140),
    }

    min_threshold = 10
    for idx in tqdm(range(total_data_cnt)):
        # 랜덤 값 생성
        threshold = random.randint(min_threshold, max_threshold)
        differences = {key: generate_random_value(0, *ranges[key], threshold) for key in ranges}
        if np.random.rand() < perfect_rate:
            for k, v in differences.items():
                differences[k] = int(np.random.uniform(-1, 1) * threshold)

        feedbacks = generate_korean_feedback(differences, threshold=threshold)

        # 낮은 값들 거르는지 여부 보고 input에서 제외
        if ignore_low_difference:
            delete_low_difference(differences, threshold)

        # input prompt를 dict으로부터 작성
        differences = {
            key.replace(" ", config.SEPARATOR): value for key, value in differences.items()
        }
        if do_numeric_to_text:
            differences['threshold'] = threshold
            input_sentence = str(numeric_to_text(differences))
        else:
            input_sentence = f"[임계치]: {threshold}\n\n"
            input_sentence += ("[입력값]: " + str(differences))

        # output sentence를 dict로부터 작성
        output_sentence = output_sentence_from_dict_simple(feedbacks)

        df["C_ID"].append(idx)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)
        df["Text"].append(input_sentence)
        df["Completion"].append(output_sentence)
    

    df = pd.DataFrame(df)
    return df

def generate_random_3D_values(ranges):
    random_values = {}
    
    for part, attributes in ranges.items():
        random_values[part] = {}
        for attr, bounds in attributes.items():
            if isinstance(bounds, dict):  # Handle closest_point_difference
                pose1 = np.random.normal(np.mean(bounds['pose1']), (bounds['pose1'][1] - bounds['pose1'][0]) / 4)
                pose2 = np.random.normal(np.mean(bounds['pose2']), (bounds['pose2'][1] - bounds['pose2'][0]) / 4)
                random_values[part][attr] = {
                    'pose1': pose1,
                    'pose2': pose2,
                    'diff': pose1 - pose2
                }
            else:
                mean = np.mean(bounds)
                std_dev = (bounds[1] - bounds[0]) / 4  # Spread around the mean
                random_values[part][attr] = np.random.normal(mean, std_dev)
    
    return random_values

def generate_int_in_bounds(bound):
    min_val, max_val = bound
    mean = (min_val + max_val) / 2
    std_dev = (max_val - min_val) / 4  # 범위의 1/4을 표준편차로 설정
    
    value = int(np.clip(np.random.normal(mean, std_dev), min_val, max_val))
    return value

def generate_float_in_bounds(bound):
    min_val, max_val = bound
    mean = (min_val + max_val) / 2
    std_dev = (max_val - min_val) / 4  # 범위의 1/4을 표준편차로 설정
    
    value = np.clip(np.random.normal(mean, std_dev), min_val, max_val)
    return value

# 범위 정의
ranges = {
    'head':{
        'lower_angle_difference': (-60, 60),
        'direction_difference': (-140, 140)
    },
    'body':{
        'bend_angle_difference': (-140, 140),
        'direction_difference': (-140, 140)
    },
    'left_arm':{
        'bend_angle_difference': (-140, 140),
        'arm_height_difference': (-140, 140),
        'hand_height_difference': (-140, 140),
        'direction_difference': (-140, 140),
        'closest_point_difference': {
            'pose1': (-0.2, 0.2),
            'pose2': (-0.5, 0.5),
            'diff': ['pose1', 'pose2']
        }
    },
    'right_arm':{
        'bend_angle_difference': (-140, 140),
        'arm_height_difference': (-140, 140),
        'hand_height_difference': (-140, 140),
        'direction_difference': (-140, 140),
        'closest_point_difference': {
            'pose1': (-0.2, 0.2),
            'pose2': (-0.5, 0.5),
            'diff': ['pose1', 'pose2']
        }
    },
    'left_leg':{
        'bend_angle_difference': (-140, 140), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
        'height_difference': (-140, 140), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
        'direction_difference': (-140, 140) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
    },
    'right_leg':{
        'bend_angle_difference': (-140, 140), # 음수면 왼팔을 더 굽혀라, 양수면 왼팔을 더 펴라
        'height_difference': (-140, 140), # 음수면 왼팔을 더 내려라, 양수면 왼팔을 더 올려라
        'direction_difference': (-140, 140) # 음수, 양수 관계없이 왼팔 방향이 맞지 않는다
    },
    'leg':{
        'knee_distance_difference': (-0.4, 0.4), # 음수면 무릎을 더 붙여라, 양수면 무릎을 너무 붙이지 마라
        'foot_distance_difference': (-0.6, 0.6)
    }
}

def refine_float_dict(differences):
    for part in differences:
            for k, v in differences[part].items():
                if isinstance(v, dict):
                    for feature in v:
                        if isinstance(v[feature], str):
                            continue

                        differences[part][k][feature] = f'{differences[part][k][feature]:.4f}'
                        differences[part][k][feature] = float(differences[part][k][feature])
                else:
                    if isinstance(ranges[part][k][0], int):
                        differences[part][k] = int(differences[part][k])
                    else:
                        differences[part][k] = f'{differences[part][k]:.4f}'
                        differences[part][k] = float(differences[part][k])
    return differences

def make_random_3D_dataset(total_data_cnt, system_prompt, angle_thres=20, dist_thres=0.12, height_thres=20, perfect_rate=0.1):
    df = {
        "System_Prompt": [], # 지시문
        "C_ID": [], #Conversation ID
        "T_ID": [], # Turn ID
        "Text": [], # 사용자가 말할 것으로 기대되는 모든 발화 내용
        "Completion": [] # CLOVA Studio가 답해야할 것으로 기대되는 모든 발화 내용
    }
    LEFT_POSITION_KEYPOINTS = [
        'left_shoulder', 'right_shoulder',
        'left_waist', 'right_waist',
        'left_hip', 'right_hip',
        'right_elbow',
        'left_knee', 'right_knee',
        'left_foot', 'right_foot',
        'belly', 'breast'
    ]

    RIGHT_POSITION_KEYPOINTS = [
        'left_shoulder', 'right_shoulder',
        'left_waist', 'right_waist',
        'left_hip', 'right_hip',
        'left_elbow',
        'left_knee', 'right_knee',
        'left_foot', 'right_foot',
        'belly', 'breast'
    ]

    for idx in tqdm(range(total_data_cnt)):
        differences = generate_random_3D_values(ranges)
        differences['left_arm']['closest_point_difference']['target_keypoint'] = random.choice(LEFT_POSITION_KEYPOINTS)
        differences['left_arm']['closest_point_difference']['user_keypoint'] = random.choice(LEFT_POSITION_KEYPOINTS)
        differences['right_arm']['closest_point_difference']['target_keypoint'] = random.choice(RIGHT_POSITION_KEYPOINTS)
        differences['right_arm']['closest_point_difference']['user_keypoint'] = random.choice(RIGHT_POSITION_KEYPOINTS)

        if np.random.rand() < perfect_rate:
            for part in differences:
                for k, v in differences[part].items():
                    if ('angle' in k or 'direction' in k):
                        differences[part][k] = generate_int_in_bounds((-angle_thres, angle_thres))
                    elif ('height' in k):
                        differences[part][k] = generate_int_in_bounds((-height_thres, height_thres))
                    elif ('distance' in k):
                        differences[part][k] = generate_float_in_bounds((-dist_thres, dist_thres))
                    else:
                        differences[part][k]['diff'] = generate_float_in_bounds((-dist_thres, dist_thres))
                        differences[part][k]['pose2'] = differences[part][k]['pose1'] - differences[part][k]['diff']


        feedback_json = pose_feedback_final.get_korean_3D_feedback(differences, angle_thres=angle_thres, dist_thres=dist_thres, height_thres=height_thres)
        agg_feedback = pose_feedback_final.aggregate_feedback(feedback_json)
        output_sentence = pose_feedback_final.get_connected_sentence_from_dict(agg_feedback)

        differences = refine_float_dict(differences)

        df["C_ID"].append(idx)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)
        df["Text"].append(str(differences))
        df["Completion"].append(output_sentence)
    
    df = pd.DataFrame(df)
    return df