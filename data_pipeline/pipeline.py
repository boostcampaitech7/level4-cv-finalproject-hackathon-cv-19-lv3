from ..dance_scoring.detector import PoseDetector, get_pose_landmark_from_detect_result
from ..dance_scoring.similarity_with_frames import *
from ..dance_scoring.util import fill_None_from_landmarks
from ..prompting.pose_compare import extract_pose_landmarks
from ..prompting.pose_feedback import json_to_prompt


def compare_video_pair(right_video_path, wrong_video_path, frame_interval=0.5):
    estimate_class = PoseDetector(model_size=2)
    right_pose_landmarker_results, right_keypoints, right_frames, right_fps = estimate_class.estimPose_video_for_dtw(right_video_path)
    right_shape = estimate_class.last_shape

    estimate_class.reset_detector()
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
            'wrong_shape': wrong_shape
        })
    
    return matched_dict_list

def get_feedback_from_keypoints(match_info_dict):
    # dictionary로부터 필요한 정보 가져오기
    right_keypoint, right_shape = match_info_dict['right_keypoint'], match_info_dict['right_shape']
    wrong_keypoint, wrong_shape = match_info_dict['wrong_keypoint'], match_info_dict['wrong_shape']

    # 사전정의된 알고리즘에 따라 관절 각도 정보를 dictionary로 가져옴
    right_pose_json = extract_pose_landmarks(right_keypoint, right_shape[1], right_shape[0])
    wrong_pose_json = extract_pose_landmarks(wrong_keypoint, wrong_shape[1], wrong_shape[0])

    # 각도 정보를 비교하여