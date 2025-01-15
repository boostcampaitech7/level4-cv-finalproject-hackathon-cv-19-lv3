import streamlit as st
import tempfile
import mediapipe_inference, util
import time
import imageio

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
    original_video_frames, only_skeleton_frames, frames = mediapipe_inference.estimPose_video(temp_filepath, thickness=10)
    new_frames = []
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