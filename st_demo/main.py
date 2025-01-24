import streamlit as st
import tempfile
import detector, util, keypoint_map, scoring
import time
import imageio
import imageio.v3 as iio
import json
import cv2
from copy import deepcopy
from util import fill_None_from_landmarks, draw_landmarks_on_image, get_closest_frame
from similarity_with_frames import l2_normalize, calculate_similarity_with_visualization, make_euclidean_similarity, make_cosine_similarity
from similarity_with_frames import get_random_pair_frames, get_all_pair_frames
from prompting.pose_compare import extract_pose_landmarks
from prompting.pose_feedback import json_to_prompt


# main title
st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Dance Pose Estimation Demo</h2>", unsafe_allow_html=True)


# sidebar
page_options = ['Single Video Pose Estimation', 'Image Compare', 'Video Compare', 'User Feedback Demo']
page_option = st.sidebar.selectbox("태스크 선택: ", page_options)
model_size = st.sidebar.slider('model_size: ', 0, 2) # 0 ~ 2 큰 숫자일수록 큰모델
seed = st.sidebar.number_input('random seed ', min_value=0, max_value=2024, step=1) #random seed
util.set_seed(seed)

# session state
if "estimate_class" not in st.session_state or (model_size != st.session_state['model_size']):
    st.session_state['model_size'] = model_size
    st.session_state['estimate_class'] = detector.PoseDetector(model_size=model_size)
if "feedback_info_1" not in st.session_state:
    st.session_state['feedback_info_1'] = None
if "feedback_info_2" not in st.session_state:
    st.session_state['feedback_info_2'] = None
if "video_names" not in st.session_state:
    st.session_state["video_names"] = ["", ""]



# start of pages
if page_option is None or page_option == page_options[0]:
    frame_option = st.sidebar.slider('frame: ', 10, 30)
    gif_options = ["only overlap", "All"]
    gif_option = st.sidebar.selectbox("예측 결과 표시 옵션: ", gif_options)
    
    # 비디오 파일 업로드
    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])
    ## gif또는 mp4로 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        gif_button = st.button("GIF 생성 및 보기")
    with col2:
        mp4_button = st.button("MP4 생성 및 보기")
    with col3:
        raw_button = st.button("frame으로부터 가져오기")

    if uploaded_file is not None and(gif_button or mp4_button or raw_button):
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filepath = temp_file.name
        

        # OpenCV로 비디오 읽고 추론
        st.info("모델로 비디오 keypoint를 추론하는 중입니다.")
        estimation_start_time = time.perf_counter()
        st.session_state['estimate_class'].reset_detector()
        original_video_frames, pose_landmarker_results = st.session_state['estimate_class'].estimPose_video(temp_filepath)

        all_landmarks = detector.get_pose_landmark_from_detect_result(pose_landmarker_results)
        all_landmarks = fill_None_from_landmarks(all_landmarks)
        all_landmarks_dict = util.landmarks_to_dict(all_landmarks)
        del all_landmarks

        estimation_end_time = time.perf_counter()
        estimation_elapsed_time = estimation_end_time - estimation_start_time
        st.success(f"pose estimation 완료. 수행시간: {estimation_elapsed_time:.2f}초")


        # 임시 파일 생성
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp_file:
            # 딕셔너리를 JSON 형식으로 임시 파일에 저장
            json.dump(all_landmarks_dict, tmp_file, indent=4)
            tmp_file_path = tmp_file.name  # 임시 파일 경로
        

        # Json 다운로드 버튼
        with open(tmp_file_path, "r") as file:
            st.download_button(
                label="Download JSON (Tempfile)",
                data=file,
                file_name="data.json",
                mime="application/json",
            )


        # 옵션에 따라 gif를 keypoint와 image가 겹쳐진 것만 보여줄지, 분리된 것도 보여줄 지 선택
        if gif_option == "only overlap":
            new_frames = [
                detector.get_overlap_from_landmarks(lm, orig) 
                for orig, lm in zip(original_video_frames, pose_landmarker_results)
            ]
        else:
            new_frames = []
            height, width = original_video_frames[0].shape[:-1]

            for orig, lm in zip(original_video_frames, pose_landmarker_results):
                original = orig
                overlap = detector.get_overlap_from_landmarks(lm, orig)
                only_skeleton = detector.get_skeleton_from_landmarks(lm, orig)

                if None in original or None in only_skeleton or None in overlap:
                    continue
                new_frames.append(util.concat_frames_with_spacing([original, only_skeleton, overlap]))
        

        # 프레임 표시 영역
        placeholder = st.empty()

        # GIF로 저장
        start_time = time.perf_counter()
        if gif_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_gif:
                gif_path = temp_gif.name
                imageio.mimsave(gif_path, new_frames, format="GIF", fps=frame_option, loop=0)  # FPS 설정
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                st.success(f"GIF 파일 생성 완료! 수행시간: {elapsed_time:.2f}초")
                st.image(gif_path, caption="생성된 GIF")
            
            # 다운로드 버튼
            with open(gif_path, "rb") as gif_file:
                gif_bytes = gif_file.read()
                st.download_button(
                    label="다운로드 GIF 파일",
                    data=gif_bytes,
                    file_name="output.gif",
                    mime="image/gif"
                )
        

        if mp4_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_mp4:
                # MP4 파일 경로
                video_path = temp_mp4.name
                iio.imwrite(video_path, new_frames, fps=frame_option, codec="libx264")
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                st.success(f"MP4 파일 생성 완료!: {elapsed_time:.2f}초")

                # Streamlit에서 재생
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            
            # 다운로드 버튼
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.download_button(
                    label="다운로드 MP4 파일",
                    data=video_bytes,
                    file_name="output.mp4",
                    mime="video/mp4"
                )

        if raw_button:
            # 비디오 플레이어 컨트롤러
            placeholder = st.empty()  # 빈 컨테이너 생성
            for frame in new_frames:
                placeholder.image(frame, channels="RGB")  # 프레임 표시
                time.sleep(1 / frame_option)  # 프레임 속도에 맞춰 대기



elif page_option == 'Image Compare':
    ignore_z = st.sidebar.slider('ignore_z: ', False, True)
    pck_thres = st.sidebar.number_input('pck_threshold', min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    # image upload
    image_1 = st.file_uploader("input_1", type=["jpg", "png", "jpeg"])
    image_2 = st.file_uploader("input_2", type=["jpg", "png", "jpeg"])

    if image_1 is not None and image_2 is not None:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file_1:
            temp_file_1.write(image_1.read())
            temp_filepath_1 = temp_file_1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file_2:
            temp_file_2.write(image_2.read())
            temp_filepath_2 = temp_file_2.name
        
        # pose estimate
        pose_landmarks_1, _, annotated_image_1, b1 = st.session_state['estimate_class'].get_detection(temp_filepath_1, landmarks_c=(234,63,247), connection_c=(117,249,77))
        pose_landmarks_2, _, annotated_image_2, b2 = st.session_state['estimate_class'].get_detection(temp_filepath_2, landmarks_c=(255, 165, 0), connection_c=(200, 200, 200))
        del _

        if pose_landmarks_1 is None or pose_landmarks_2 is None:
            raise ValueError("each image has to at least one person in each of them")

        # annotated image 각각 표시
        col1, col2 = st.columns(2)
        with col1:
            st.image(annotated_image_1)
        with col2:
            st.image(annotated_image_2)
        
        # normalize한 경우에 대한 overlap image와 점수 계산
        pose_landmarks_np_1 = scoring.refine_landmarks(pose_landmarks_1)
        pose_landmarks_np_2 = scoring.refine_landmarks(pose_landmarks_2)
        evaluation_results = scoring.evaluate_everything(pose_landmarks_np_1, b1, pose_landmarks_np_2, b2, pck_thres=pck_thres, normalize=True, ignore_z=ignore_z)

        overlap_img1 = cv2.cvtColor(cv2.imread(temp_filepath_1), cv2.COLOR_BGR2RGB)
        overlap_img1 = util.image_alpha_control(overlap_img1, alpha=0.4)
        overlap_img1 = util.draw_landmarks_on_image(overlap_img1, pose_landmarks_1)

        normalized_pose_landmarks_2 = deepcopy(pose_landmarks_2)
        normalized_pose_landmarks_np_2 = scoring.normalize_landmarks_to_range(
            scoring.refine_landmarks(pose_landmarks_1, target_keys=keypoint_map.TOTAL_KEYPOINTS), 
            scoring.refine_landmarks(pose_landmarks_2, target_keys=keypoint_map.TOTAL_KEYPOINTS)
        )

        for i, landmarks in enumerate(normalized_pose_landmarks_2):
            landmarks.x = normalized_pose_landmarks_np_2[i, 0]
            landmarks.y = normalized_pose_landmarks_np_2[i, 1]
            landmarks.z = normalized_pose_landmarks_np_2[i, 2]
        overlap_img1 = util.draw_landmarks_on_image(overlap_img1, normalized_pose_landmarks_2, landmarks_c=(255, 165, 0), connection_c=(200, 200, 200))

        st.subheader("Normalize를 적용시의 결과: ")
        col3, col4 = st.columns(2)
        with col3:
            st.json(evaluation_results)
        with col4:
            st.image(overlap_img1)


        # normalize를 진행하지 않은 경우에 대한 overlap image와 점수 계산
        evaluation_results_2 = scoring.evaluate_everything(pose_landmarks_np_1, b1, scoring.refine_landmarks(pose_landmarks_2), b2, pck_thres=pck_thres, normalize=False, ignore_z=ignore_z)
        overlap_img2 = cv2.cvtColor(cv2.imread(temp_filepath_1), cv2.COLOR_BGR2RGB)
        overlap_img2 = util.image_alpha_control(overlap_img2, alpha=0.4)
        overlap_img2 = util.draw_landmarks_on_image(overlap_img2, pose_landmarks_1)
        overlap_img2 = util.draw_landmarks_on_image(overlap_img2, pose_landmarks_2, landmarks_c=(255, 165, 0), connection_c=(200, 200, 200))


        st.subheader("Normalize를 적용하지 않을 시의 결과: ")
        col5, col6 = st.columns(2)
        with col5:
            st.json(evaluation_results_2)
        with col6:
            st.image(overlap_img2)


elif page_option=="Video Compare":
    frame_option = st.sidebar.slider('frame: ', 10, 30) # 보여질 동영상의 프레임 설정
    ignore_z = st.sidebar.slider('ignore_z: ', False, True) # pose기반 difference 계산 시 z좌표를 사용할지 여부
    use_dtw = st.sidebar.slider('use_dtw_to_calculate_video_sim: ', False, True) # frame간 점수계산을 위한 매칭에서 dtw를 사용할지 여부
    pck_thres = st.sidebar.number_input('pck_threshold', min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    # 비디오 파일 업로드
    video_1 = st.file_uploader("video_1", type=["mp4", "mov", "avi", "mkv"])
    video_2 = st.file_uploader("video_2", type=["mp4", "mov", "avi", "mkv"])
    

    if video_1 and video_2:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_1:
            temp_file_1.write(video_1.read())
            temp_filepath_1 = temp_file_1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_2:
            temp_file_2.write(video_2.read())
            temp_filepath_2 = temp_file_2.name
        
        # 1번 비디오 landmarks 추출
        st.session_state['estimate_class'].reset_detector()
        original_video_frames_1, pose_landmarker_results_1 = st.session_state['estimate_class'].estimPose_video(temp_filepath_1)
        all_landmarks_1 = fill_None_from_landmarks(detector.get_pose_landmark_from_detect_result(pose_landmarker_results_1))
        height_1, width_1 = st.session_state['estimate_class'].last_shape

        # 2번 비디오 landmarks 추출
        st.session_state['estimate_class'].reset_detector() # timestamp를 초기화해야 다음 동영상 분석 가능
        original_video_frames_2, pose_landmarker_results_2 = st.session_state['estimate_class'].estimPose_video(temp_filepath_2)
        all_landmarks_2 = fill_None_from_landmarks(detector.get_pose_landmark_from_detect_result(pose_landmarker_results_2))
        height_2, width_2 = st.session_state['estimate_class'].last_shape


        total_results, low_score_frames = scoring.get_score_from_frames(
            all_landmarks_1, all_landmarks_2, pck_thres=pck_thres, thres=0.4, ignore_z=ignore_z, use_dtw=use_dtw
        )
        for k, v in total_results.items():
            if "matched" in k: continue
            s = f"{k}: {v}"
            print(s)
            st.write(s)


        matched = total_results['matched']
        matched_frame_list = total_results['matched_frame']

        ann_1 = [
            detector.get_overlap_from_landmarks(lm, orig) 
            for orig, lm in zip(original_video_frames_1, pose_landmarker_results_1)
        ]
        ann_2 = [
            detector.get_overlap_from_landmarks(lm, orig) 
            for orig, lm in zip(original_video_frames_2, pose_landmarker_results_2)
        ]
        for i, (frame_num_1, frame_num_2) in enumerate(matched_frame_list):
            match_dict = matched[i]

            matched_key_list = [keypoint_map.REVERSE_KEYPOINT_MAPPING[k] for k in match_dict.keys() if match_dict[k]]
            frame_1_landmarks = all_landmarks_1[frame_num_1]
            frame_2_landmarks = all_landmarks_2[frame_num_2]
            if frame_1_landmarks is None or frame_2_landmarks is None:
                continue

            for k in matched_key_list:
                x1, y1 = frame_1_landmarks[k].x, frame_1_landmarks[k].y 
                x2, y2 = frame_2_landmarks[k].x, frame_2_landmarks[k].y 
                util.draw_circle_on_image(ann_1[frame_num_1], x1, y1, r=5)
                util.draw_circle_on_image(ann_2[frame_num_2], x2, y2, r=5)
        
        new_frames = []
        max_frame_height = util.get_max_height_from_frames([ann_1[0], ann_2[0]])
        for f1, f2 in zip(ann_1, ann_2):
            new_frames.append(util.concat_frames_with_spacing([f1, f2], max_frame_height))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_mp4:
            # MP4 파일 경로
            video_path = temp_mp4.name
            iio.imwrite(video_path, new_frames, fps=frame_option, codec="libx264")
            end_time = time.perf_counter()
            st.success("MP4 파일 생성 완료!")

            # Streamlit에서 재생
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
else:
    # 비디오 파일 업로드
    video_1 = st.file_uploader("video_1", type=["mp4", "mov", "avi", "mkv"])
    video_2 = st.file_uploader("video_2", type=["mp4", "mov", "avi", "mkv"])

    # estimation 수행
    if video_1 and video_2:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_1:
            temp_file_1.write(video_1.read())
            temp_filepath_1 = temp_file_1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_2:
            temp_file_2.write(video_2.read())
            temp_filepath_2 = temp_file_2.name
        
        if st.session_state["video_names"][0] != video_1.name or st.session_state['feedback_info_1'] is None:
            st.session_state['estimate_class'].reset_detector()
            pose_landmarker_results1, keypoints1, frames1, fps1 = st.session_state['estimate_class'].estimPose_video_for_dtw(temp_filepath_1)
            height1, width1 = st.session_state['estimate_class'].last_shape
            # session state 설정
            st.session_state["video_names"][0] = video_1.name
            st.session_state['feedback_info_1'] = (pose_landmarker_results1, keypoints1, frames1, fps1, height1, width1)
        else:
            pose_landmarker_results1, keypoints1, frames1, fps1, height1, width1 = st.session_state['feedback_info_1']
        
        if st.session_state["video_names"][1] != video_2.name or st.session_state['feedback_info_2'] is None:
            st.session_state['estimate_class'].reset_detector()
            pose_landmarker_results2, keypoints2, frames2, fps2 = st.session_state['estimate_class'].estimPose_video_for_dtw(temp_filepath_2)
            height2, width2 = st.session_state['estimate_class'].last_shape
            # session state 설정
            st.session_state["video_names"][1] = video_2.name
            st.session_state['feedback_info_2'] = (pose_landmarker_results2, keypoints2, frames2, fps2, height2, width2)
        else:
            pose_landmarker_results2, keypoints2, frames2, fps2, height2, width2 = st.session_state['feedback_info_2']

        # keypoints L2 정규화
        keypoints1 = l2_normalize(keypoints1)
        keypoints2 = l2_normalize(keypoints2)

        # 유사도 및 시각화 데이터 계산
        distance, average_cosine_similarity, average_euclidean_distance, average_oks, average_pck, pairs = calculate_similarity_with_visualization(
            keypoints1, keypoints2
        )

        # HTML과 CSS를 사용해 배경색 부여
        st.markdown(
            """
            <style>
            .score-card {
                background-color: #6e65c2; /* 배경색 */
                padding: 20px;
                border-radius: 10px; /* 모서리 둥글게 */
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* 그림자 효과 */
                margin-bottom: 20px;
            }
            .score-title {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                text-align: center;
            }
            .metric {
                font-size: 18px;
                margin: 10px 0;
                text-align: left;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # HTML 성적표 동적 생성
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">성적표</div>
                <div class="metric">📌 Average Cosine Similarity: <b>{make_cosine_similarity(average_cosine_similarity):.2f}</b></div>
                <div class="metric">📌 Average Euclidean Similarity: <b>{make_euclidean_similarity(average_euclidean_distance):.2f}</b></div>
                <div class="metric">📌 Average OKS: <b>{average_oks:.2f}</b></div>
                <div class="metric">📌 Average PCK: <b>{average_pck:.2f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        random_matched_list = []
        for idx2, frame in enumerate(frames2):
            idx1 = get_random_pair_frames(pairs, idx2)
            random_matched_list.append(idx1)
            
            landmarks1 = pose_landmarker_results1[idx1].pose_landmarks[0]
            landmarks2 = pose_landmarker_results2[idx2].pose_landmarks[0]
            normalized_pose_landmarks_np_1 = scoring.normalize_landmarks_to_range(
                scoring.refine_landmarks(landmarks2, target_keys=keypoint_map.TOTAL_KEYPOINTS), 
                scoring.refine_landmarks(landmarks1, target_keys=keypoint_map.TOTAL_KEYPOINTS)
            )
            
            for i, landmarks in enumerate(pose_landmarker_results1[idx1].pose_landmarks[0]):
                landmarks.x = normalized_pose_landmarks_np_1[i, 0]
                landmarks.y = normalized_pose_landmarks_np_1[i, 1]
                landmarks.z = normalized_pose_landmarks_np_1[i, 2]
            frames2[idx2] = draw_landmarks_on_image(frame, pose_landmarker_results1[idx1])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_mp4:
            # MP4 파일 경로
            video_path = temp_mp4.name
            iio.imwrite(video_path, frames2, fps=fps2, codec="libx264")
            end_time = time.perf_counter()
            st.success("MP4 파일 생성 완료!")

            # Streamlit에서 재생
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
        
        total_frame_len = len(frames2)
        total_time = int(total_frame_len / fps2)

        st.title("피드백을 받을 시간을 선택해주세요")
        # 슬라이더 추가
        # 슬라이더와 버튼을 폼 내부에 배치
        with st.form("time_selection_form"):
            st.title("Select a Time")
            target_time = st.slider(
                "Select a value:",       # Slider label
                min_value=1,             # Minimum value
                max_value=total_time,    # Maximum value
                value=(total_time + 1) // 2  # Default value
            )
            # 확인 버튼
            submit_button = st.form_submit_button("Confirm")

        if submit_button:
            user_idx = get_closest_frame(target_time, total_frame_len, fps2)
            user_landmark = pose_landmarker_results2[user_idx]
            target_idx = random_matched_list[user_idx]
            target_landmark = pose_landmarker_results1[target_idx]


            # 슬라이더 값을 기반으로 프레임 계산
            target_frame = (target_time * fps2) if target_time else 0
            st.write(f"선택된 시간: {target_time}초")
            st.write(f"해당 프레임: {target_frame}")

            user_pose_landmarks_json = extract_pose_landmarks(user_landmark, width2, height2)
            target_pose_landmarks_json = extract_pose_landmarks(target_landmark, width1, height1)
            feedback = json_to_prompt(target_pose_landmarks_json, user_pose_landmarks_json)

            col1, col2 = st.columns(2)
            with col1:
                st.json(feedback)
            with col2:
                st.image(frames2[user_idx])
