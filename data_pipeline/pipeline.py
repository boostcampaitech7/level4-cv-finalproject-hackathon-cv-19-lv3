import sys
import random
from tqdm import tqdm
sys.path.append("./")
import pandas as pd
import numpy as np

from dance_scoring.detector import PoseDetector, post_process_pose_landmarks
from dance_scoring.similarity_with_frames import *
from dance_feedback.pose_compare import extract_3D_pose_landmarks
from dance_feedback.pose_feedback import *



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


def get_feedback_from_keypoints(match_info_dict):
    # dictionary로부터 필요한 정보 가져오기
    right_keypoint, right_shape = match_info_dict['right_keypoint'], match_info_dict['right_shape']
    wrong_keypoint, wrong_shape = match_info_dict['wrong_keypoint'], match_info_dict['wrong_shape']

    # 사전정의된 알고리즘에 따라 관절 각도 정보를 dictionary로 가져옴
    right_pose_json = extract_3D_pose_landmarks(right_keypoint)
    wrong_pose_json = extract_3D_pose_landmarks(wrong_keypoint)

    # 각도 정보를 비교하여 수치적인 차이와 그에 해당하는 자연어 피드백을 dictionary형태로 가져옴
    differences = get_difference_dict(right_pose_json, wrong_pose_json)
    return differences


def make_dataset(matched_dict_list, system_prompt, start_CID=0, angle_thres=20, dist_thres=0.12, height_thres=20):
    df = {
        "System_Prompt": [], # 지시문
        "C_ID": [], #Conversation ID
        "T_ID": [], # Turn ID
        "Text": [], # 사용자가 말할 것으로 기대되는 모든 발화 내용
        "Completion": [] # CLOVA Studio가 답해야할 것으로 기대되는 모든 발화 내용
    }

    for idx, matched_dict in enumerate(matched_dict_list):
        differences = refine_float_dict(get_feedback_from_keypoints(matched_dict))
        feedbacks = get_korean_3D_feedback(differences, angle_thres=20, dist_thres=0.12, height_thres=20)
        feedbacks = aggregate_feedback(feedbacks)

        # output sentence를 dict로부터 작성
        output_sentence = get_connected_sentence_from_dict(feedbacks)

        df["C_ID"].append(idx+start_CID)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)
        df["Text"].append(str(differences))
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
        'bend_angle_difference': (-140, 140),
        'height_difference': (-140, 140),
        'direction_difference': (-140, 140)
    },
    'right_leg':{
        'bend_angle_difference': (-140, 140), 
        'height_difference': (-140, 140), 
        'direction_difference': (-140, 140) 
    },
    'leg':{
        'knee_distance_difference': (-0.4, 0.4),
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


        feedback_json = get_korean_3D_feedback(differences, angle_thres=angle_thres, dist_thres=dist_thres, height_thres=height_thres)
        agg_feedback = aggregate_feedback(feedback_json)
        output_sentence = get_connected_sentence_from_dict(agg_feedback)
        differences = refine_float_dict(differences)

        df["C_ID"].append(idx)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)
        df["Text"].append(str(differences))
        df["Completion"].append(output_sentence)
    
    df = pd.DataFrame(df)
    return df