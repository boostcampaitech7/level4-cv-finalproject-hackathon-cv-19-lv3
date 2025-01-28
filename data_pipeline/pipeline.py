from ..st_demo.detector import PoseDetector, get_pose_landmark_from_detect_result
from ..st_demo.similarity_with_frames import *
from ..st_demo.util import fill_None_from_landmarks


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