import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import tempfile
import mediapipe_inference, util
import time
import imageio
import imageio.v3 as iio
import cv2
import av

# main title
st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Dance Pose Estimation Demo</h2>", unsafe_allow_html=True)


# sidebar
page_options = ['Single Video Pose Estimation', 'Record and Compare']
page_option = st.sidebar.selectbox("태스크 선택: ", page_options)
frame_option = st.sidebar.slider('frame: ', 10, 30)


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


if page_option is None or page_option == page_options[0]:
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
            st.session_state.original_video_frames, st.session_state.only_skeleton_frames, st.session_state.frames = mediapipe_inference.estimPose_video(temp_filepath, thickness=5)
    
        if st.session_state['estimation_end_time'] is None:
            st.session_state.estimation_end_time = time.perf_counter()
        st.session_state.estimation_elapsed_time = st.session_state['estimation_end_time'] - st.session_state['estimation_start_time']
        st.success(f"pose estimation 완료. 수행시간: {st.session_state['estimation_elapsed_time']:.2f}초")

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
    # 녹화 데이터 저장을 위한 리스트
    frames = []

    # VideoProcessor 클래스 정의
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.recording = False

        def recv(self, frame):
            # WebRTC에서 전달받은 프레임 처리
            img = frame.to_ndarray(format="bgr24")
            
            # 녹화 중이면 프레임 저장
            if self.recording:
                frames.append(img)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # WebRTC 스트리머 초기화
    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 녹화 컨트롤 버튼
    if ctx.video_processor:
        if st.button("녹화 시작"):
            ctx.video_processor.recording = True
            frames.clear()  # 녹화 시작 시 이전 데이터 삭제
            st.info("녹화를 시작했습니다.")
        
        if st.button("녹화 중지"):
            ctx.video_processor.recording = False
            st.success("녹화를 중지했습니다.")

            # 녹화된 비디오 저장
            if frames:
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
                    height, width, _ = frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(video_file.name, fourcc, 20, (width, height))

                    for frame in frames:
                        out.write(frame)
                    out.release()
                
                    st.success(f"녹화된 비디오가 저장되었습니다: {video_file.name}")
                    st.video(video_file.name)

    # WebRTC 상태 정보 출력
    if ctx.state.playing:
        st.text("웹캠이 활성화되었습니다.")