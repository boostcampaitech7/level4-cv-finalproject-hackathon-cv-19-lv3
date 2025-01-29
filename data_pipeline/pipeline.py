import sys
sys.path.append("./")

from dance_scoring.detector import PoseDetector, get_pose_landmark_from_detect_result
from dance_scoring.similarity_with_frames import *
from dance_scoring.util import fill_None_from_landmarks
from prompting.pose_compare import extract_pose_landmarks
from prompting.pose_feedback import json_to_prompt, generate_feedback
import pandas as pd


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
    right_pose_landmarker_results, right_keypoints, right_frames, right_fps = estimate_class.estimPose_video_for_dtw(right_video_path)
    right_shape = estimate_class.last_shape

    wrong_pose_landmarker_results, wrong_keypoints, wrong_frames, wrong_fps = estimate_class.estimPose_video_for_dtw(wrong_video_path)
    wrong_shape = estimate_class.last_shape
    
    # keypoints L2 정규화
    right_keypoints = l2_normalize(right_keypoints)
    wrong_keypoints = l2_normalize(wrong_keypoints)

    # 유사도 및 시각화 데이터 계산
    distance, average_cosine_similarity, average_euclidean_distance, average_oks, average_pck, pairs = calculate_similarity_with_visualization(
        right_keypoints, wrong_keypoints
    )

    # keypoint 결과 저장하기 편하도록 정제하는 과정
    right_pose_landmarker_results = fill_None_from_landmarks(get_pose_landmark_from_detect_result(right_pose_landmarker_results))
    wrong_pose_landmarker_results = fill_None_from_landmarks(get_pose_landmark_from_detect_result(wrong_pose_landmarker_results))
    
    # 매치된 pair끼리 frame, keypoint 저장
    matched_dict_list = []
    for idx1, frame in enumerate(right_frames):
        if idx1 % (right_fps * frame_interval) != 0:
            continue

        idx2 = get_center_pair_frames(pairs, idx1, matched_idx=0)
        matched_dict_list.append({
            'right_idx': idx1,
            'wrong_idx': idx2,
            'time': right_fps * idx1,
            'right_frame': frame,
            'wrong_frame': wrong_frames[idx2],
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



def get_feedback_from_keypoints(match_info_dict, feedback_thres = 30):
    # dictionary로부터 필요한 정보 가져오기
    right_keypoint, right_shape = match_info_dict['right_keypoint'], match_info_dict['right_shape']
    wrong_keypoint, wrong_shape = match_info_dict['wrong_keypoint'], match_info_dict['wrong_shape']

    # 사전정의된 알고리즘에 따라 관절 각도 정보를 dictionary로 가져옴
    right_pose_json = extract_pose_landmarks(right_keypoint, right_shape[1], right_shape[0])
    wrong_pose_json = extract_pose_landmarks(wrong_keypoint, wrong_shape[1], wrong_shape[0])

    # 각도 정보를 비교하여 수치적인 차이와 그에 해당하는 자연어 피드백을 dictionary형태로 가져옴
    differences, _ = json_to_prompt(right_pose_json, wrong_pose_json)
    feedbacks = generate_feedback(differences, threshold=feedback_thres)
    return differences, feedbacks


def numeric_to_text(numeric_result_json):
    for k, v in numeric_result_json.items():
        if v == 0:
            numeric_result_json[k] = "목표 자세와 차이가 없습니다."
        else:
            if 'angle' in k:
                numeric_result_json[k] = f"목표 자세에 대해 {v}만큼의 각도 차이가 있습니다."
            else:
                numeric_result_json[k] = f"목표 자세에 대해 {v}만큼의 높이 차이가 있습니다."
    return numeric_result_json


def make_dataset(matched_dict_list, system_prompt, start_CID=0):
    df = {
        "System_Prompt": [], # 지시문
        "C_ID": [], #Conversation ID
        "T_ID": [], # Turn ID
        "Text": [], # 사용자가 말할 것으로 기대되는 모든 발화 내용
        "Completion": [] # CLOVA Studio가 답해야할 것으로 기대되는 모든 발화 내용
    }

    for idx, matched_dict in enumerate(matched_dict_list):
        df["C_ID"].append(idx+start_CID)
        df["T_ID"].append(0)
        df["System_Prompt"].append(system_prompt)

        differences, feedbacks = get_feedback_from_keypoints(matched_dict)
        input_sentence = ""
        output_sentence = "자세 차이를 기반으로 피드백을 드리도록 하겠습니다.\n"

        differences = numeric_to_text(differences)
        for k, v in differences.items():
            input_sentence += f"{k}: {v}\n"
        
        for k, v in feedbacks.items():
            output_sentence += f"{english_to_korean[k] if k in english_to_korean else k}: {v}\n"
        
        if "perfect_msg" not in output_sentence:
            output_sentence += "나머지 자세는 모두 완벽합니다! 앞으로도 함께 노력해봐요!"
        else:
            output_sentence += "대단해요!"

        df["Text"].append(input_sentence)
        df["Completion"].append(output_sentence)
    
    df = pd.DataFrame(df)
    return df