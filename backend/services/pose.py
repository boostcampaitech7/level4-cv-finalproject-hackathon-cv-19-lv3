import os
import json
import cv2
from fastapi.responses import JSONResponse
from models.mediapipe import mp_model

VIDEO_FILES = {"origin": ("video.mp4", "video.json"), "user": ("user.mp4", "user.json")}
EXCEPTIONS = [1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22]
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

        if result and result.pose_landmarks.landmark:
            points = [
                (round(lm.x, 4), round(lm.y, 4))
                for i, lm in enumerate(result.pose_landmarks.landmark)
                if i not in EXCEPTIONS
            ]
        else:
            points = [(-1, -1)] * (33 - len(EXCEPTIONS))

        all_frames_points.append(points)

    cap.release()
    return fps, width, height, all_frames_points

async def extract_pose_from_video(folder_id: str):
    try:
        root_path = os.path.join("data", folder_id)
        video_file, json_file = VIDEO_FILES.get("origin", VIDEO_FILES["origin"])
        video_path = os.path.join(root_path, video_file)
        json_path = os.path.join(root_path, json_file)

        fps, width, height, all_frames_points = extract_pose(video_path)

        data = {"fps": fps, "width": width, "height": height, "all_frames_points": all_frames_points}
        with open(json_path, "w") as f:
            json.dump(data, f)

        return JSONResponse(content={"message": "Success"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed: {str(e)}"}, status_code=400)

async def extract_user_pose(folder_id: str, video):
    try:
        root_path = os.path.join("data", folder_id)
        video_file, json_file = VIDEO_FILES.get("user", VIDEO_FILES["user"])
        video_path = os.path.join(root_path, video_file)
        json_path = os.path.join(root_path, json_file)

        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())

        fps, width, height, all_frames_points = extract_pose(video_path)

        data = {"fps": fps, "width": width, "height": height, "all_frames_points": all_frames_points}
        with open(json_path, "w") as f:
            json.dump(data, f)

        return JSONResponse(content={"message": "Success"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed: {str(e)}"}, status_code=400)