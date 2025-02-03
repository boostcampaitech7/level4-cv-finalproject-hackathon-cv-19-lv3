import os
import json
import cv2
import mediapipe as mp
from fastapi.responses import JSONResponse
from models.mediapipe import mp_model

ORIGIN_VIDEO = "video.mp4"
ORIGIN_JSON = "video.json"
USER_VIDEO = "user.mp4"
USER_JSON = "user.json"

EXCEPTIONS = [1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22]
detector = mp_model().get_detector()

def extract_points(video_path: str):
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = detector.detect(mp_image)

        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            points = [(round(lm.x, 4), round(lm.y, 4)) for i, lm in enumerate(landmarks) if i not in EXCEPTIONS]
            all_frames_points.append(points)

    cap.release()
    return fps, width, height, all_frames_points


async def extract_pose_from_video(folder_id: str):
    try:
        root_path = os.path.join("data", folder_id)
        video_path = os.path.join(root_path, ORIGIN_VIDEO)

        # YouTube video 모든 프레임 pose extract
        fps, width, height, all_frames_points = extract_points(video_path)
        data = {"fps": fps, "width": width, "height": height, "all_frames_points": all_frames_points}

        # pose extract 정보를 저장
        with open(os.path.join(root_path, ORIGIN_JSON), "w") as f:
            json.dump(data, f)

        return JSONResponse(content={"message": "Success save pose"}, status_code=200)
    except Exception:
        return JSONResponse(content={"error": "Failed save pose"}, status_code=400)

async def extract_user_pose(folder_id, video):
    try:
        root_path = os.path.join("data", folder_id)
        video_path = os.path.join(root_path, USER_VIDEO)

        # user video를 저장
        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # user video 모든 프레임 pose extract
        fps, width, height, all_frames_points = extract_points(video_path)
        data = {"fps": fps, "width": width, "height": height, "all_frames_points": all_frames_points}

        # pose extract 정보를 저장
        with open(os.path.join(root_path, USER_JSON), "w") as f:
            json.dump(data, f)

        return JSONResponse(content={"message": "Success save pose"}, status_code=200)
    except Exception:
        return JSONResponse(content={"error": "Failed save pose"}, status_code=400)
