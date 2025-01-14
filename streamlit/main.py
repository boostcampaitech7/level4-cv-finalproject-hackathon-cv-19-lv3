import streamlit as st
import tempfile
import mediapipe_inference
import time

st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Dance Pose Estimation Demo</h2>", unsafe_allow_html=True)



# 비디오 파일 업로드
uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filepath = temp_file.name
    
    # OpenCV로 비디오 읽기
    frames = mediapipe_inference.estimPose_video(temp_filepath)

    # 프레임 표시 영역
    placeholder = st.empty()

    # 재생 속도 조정 (초당 프레임)
    fps = st.slider("FPS (초당 프레임)", 1, 30, 10)
    # 재생 버튼
    if st.button("재생"):
        # frames 리스트를 순차적으로 표시
        for i, frame in enumerate(frames):
            # placeholder에 이미지 표시
            placeholder.image(frame, channels="RGB", caption=f"Frame {i+1}/{len(frames)}")
            
            # FPS에 따라 지연
            time.sleep(1 / fps)