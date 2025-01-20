import streamlit as st
import tempfile
import detector, util, keypoint_map, scoring
import time
import imageio
import imageio.v3 as iio
import json

# main title
st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Dance Pose Estimation Demo</h2>", unsafe_allow_html=True)


# sidebar
page_options = ['Single Video Pose Estimation', 'Image Compare']
page_option = st.sidebar.selectbox("태스크 선택: ", page_options)
frame_option = st.sidebar.slider('frame: ', 10, 30)
model_size = st.sidebar.slider('model_size: ', 0, 2)
seed = st.sidebar.number_input('random seed ', min_value=0, max_value=2024, step=1)


# session state
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "original_video_frames" not in st.session_state:
    st.session_state["original_video_frames"] = None
if "only_skeleton_frames" not in st.session_state:
    st.session_state["only_skeleton_frames"] = None
if "frames" not in st.session_state:
    st.session_state["frames"] = None
if "estimation_start_time" not in st.session_state:
    st.session_state["estimation_start_time"] = None
if "estimation_end_time" not in st.session_state:
    st.session_state["estimation_end_time"] = None
if "all_landmarks" not in st.session_state:
    st.session_state["all_landmarks"] = None
if "all_landmarks_dict" not in st.session_state:
    st.session_state["all_landmarks_dict"] = None
if "estimate_class" not in st.session_state:
    st.session_state['estimate_class'] = detector.PoseDetector(model_size=model_size)


if page_option is None or page_option == page_options[0]:
    util.set_seed(seed)
    gif_options = ["only overlap", "All"]
    gif_option = st.sidebar.selectbox("예측 결과 표시 옵션: ", gif_options)
    
    # 비디오 파일 업로드
    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filepath = temp_file.name
        
        # OpenCV로 비디오 읽고 추론
        if st.session_state['estimation_start_time'] is None:
            st.session_state.estimation_start_time = time.perf_counter()
        if st.session_state["uploaded_file"] is None or uploaded_file.name != st.session_state["uploaded_file"].name:
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state.original_video_frames, st.session_state.only_skeleton_frames, st.session_state.frames, st.session_state.all_landmarks = st.session_state['estimate_class'].estimPose_video(temp_filepath, thickness=5)
            st.session_state['all_landmarks_dict'] = util.landmarks_to_dict(st.session_state['all_landmarks'])
    
        if st.session_state['estimation_end_time'] is None:
            st.session_state.estimation_end_time = time.perf_counter()
        st.session_state.estimation_elapsed_time = st.session_state['estimation_end_time'] - st.session_state['estimation_start_time']
        st.success(f"pose estimation 완료. 수행시간: {st.session_state['estimation_elapsed_time']:.2f}초")
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp_file:
            # 딕셔너리를 JSON 형식으로 임시 파일에 저장
            json.dump(st.session_state['all_landmarks_dict'], tmp_file, indent=4)
            tmp_file_path = tmp_file.name  # 임시 파일 경로
            
        # Streamlit 다운로드 버튼 추가
        with open(tmp_file_path, "r") as file:
            st.download_button(
                label="Download JSON (Tempfile)",
                data=file,
                file_name="data.json",
                mime="application/json",
            )
            
        

        new_frames = []
        # 옵션에 따라 gif를 keypoint와 image가 겹쳐진 것만 보여줄지, 분리된 것도 보여줄 지 선택
        if gif_option == "only overlap":
            new_frames = st.session_state["frames"]
        else:
            for i in range(len(st.session_state["frames"])):
                original = st.session_state["original_video_frames"][i]
                overlap = st.session_state["frames"][i]
                only_skeleton = st.session_state["only_skeleton_frames"][i]

                new_frames.append(util.concat_frames_with_spacing([original, only_skeleton, overlap]))


        ## gif또는 mp4로 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            gif_button = st.button("GIF 생성 및 보기")
        with col2:
            mp4_button = st.button("MP4 생성 및 보기")
        with col3:
            raw_button = st.button("frame으로부터 가져오기")
        

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
                iio.imwrite(video_path, new_frames, fps=frame_option)
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


else:
    image_1 = st.file_uploader("input_1", type=["jpg", "png", "jpeg"])
    image_2 = st.file_uploader("input_2", type=["jpg", "png", "jpeg"])

    if image_1 is not None and image_2 is not None:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_1:
            temp_file_1.write(image_1.read())
            temp_filepath_1 = temp_file_1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_2:
            temp_file_2.write(image_2.read())
            temp_filepath_2 = temp_file_2.name
        
        pose_landmarks_1, segmentation_masks_1, annotated_image_1, b1 = st.session_state['estimate_class'].get_detection(temp_filepath_1)
        pose_landmarks_2, segmentation_masks_2, annotated_image_2, b2 = st.session_state['estimate_class'].get_detection(temp_filepath_2)

        col1, col2 = st.columns(2)
        with col1:
            st.image(annotated_image_1)
        with col2:
            st.image(annotated_image_2)
        
        pose_landmarks_np_1 = scoring.refine_landmarks(pose_landmarks_1)
        pose_landmarks_np_2 = scoring.refine_landmarks(pose_landmarks_2)

        evaluation_results = scoring.evaluate_everything(pose_landmarks_np_1, b1, pose_landmarks_np_2, b2, normalize=True)
        st.subheader("Normalize를 적용시의 결과: ")
        st.json(evaluation_results)

        evaluation_results_2 = scoring.evaluate_everything(pose_landmarks_np_1, b1, scoring.refine_landmarks(pose_landmarks_2), b2, normalize=False)
        st.subheader("Normalize를 적용하지 않을 시의 결과: ")
        st.json(evaluation_results_2)