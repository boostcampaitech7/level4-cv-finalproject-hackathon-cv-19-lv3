import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import tempfile
import mediapipe_inference, util
import time
import imageio

st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Dance Pose Estimation Demo</h2>", unsafe_allow_html=True)

page_options = ['Single Video Pose Estimation', 'Record and Compare']
page_option = st.sidebar.selectbox("태스크 선택: ", page_options)


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
        
        # OpenCV로 비디오 읽기
        original_video_frames, only_skeleton_frames, frames = mediapipe_inference.estimPose_video(temp_filepath, thickness=5)
        new_frames = []

        if gif_option == "only overlap":
            new_frames = frames
        else:
            for i in range(len(frames)):
                original = original_video_frames[i]
                overlap = frames[i]
                only_skeleton = only_skeleton_frames[i]

                new_frames.append(util.concat_frames_with_spacing([original, only_skeleton, overlap]))

        # 프레임 표시 영역
        placeholder = st.empty()

        # GIF로 저장
        if st.button("GIF 생성 및 보기"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_gif:
                gif_path = temp_gif.name
                imageio.mimsave(gif_path, new_frames, format="GIF", fps=30, loop=0)  # FPS 설정
                st.success("GIF 파일 생성 완료!")
                
                # GIF 표시
                st.image(gif_path, caption="생성된 GIF")
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
                import cv2
                import tempfile

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