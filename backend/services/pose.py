import os
import cv2
import h5py
from fastapi.responses import JSONResponse
from constants import FilePaths
from models.mediapipe import mp_model

SELECTED_POINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
pose = mp_model()

def extract_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_frames_points = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.get_result(frame_rgb)

        if result.pose_landmarks:
            points = [
                (round(lm.x, 4), round(lm.y, 4), round(lm.z, 4))
                for i, lm in enumerate(result.pose_world_landmarks.landmark)
                if i in SELECTED_POINTS
            ]
        else:
            points = [(-1, -1, -1)] * len(SELECTED_POINTS)

        all_frames_points.append(points)

    cap.release()
    return fps, width, height, all_frames_points

async def extract_pose_from_video(folder_id: str):
    try:
        root_path = os.path.join("data", folder_id)
        video_file, h5_file = FilePaths.ORIGIN_MP4.value, FilePaths.ORIGIN_H5.value
        video_path = os.path.join(root_path, video_file)
        h5_path = os.path.join(root_path, h5_file)

        fps, width, height, all_frames_points = extract_pose(video_path)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("fps", data=fps)
            f.create_dataset("width", data=width)
            f.create_dataset("height", data=height)
            f.create_dataset("all_frames_points", data=all_frames_points, compression="gzip")
        return JSONResponse(content={"message": "Success"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed: {str(e)}"}, status_code=400)
    
async def extract_user_pose(folder_id: str, video):
    try:
        root_path = os.path.join("data", folder_id)
        video_file, h5_file = FilePaths.USER_MP4.value, FilePaths.USER_H5.value
        video_path = os.path.join(root_path, video_file)
        h5_path = os.path.join(root_path, h5_file)

        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())

        fps, width, height, all_frames_points = extract_pose(video_path)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("fps", data=fps)
            f.create_dataset("width", data=width)
            f.create_dataset("height", data=height)
            f.create_dataset("all_frames_points", data=all_frames_points, compression="gzip")
        return JSONResponse(content={"message": "Success"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed: {str(e)}"}, status_code=400)
