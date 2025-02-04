import os
import json
from fastapi.responses import JSONResponse

ORIGIN_JSON = "video.json"
USER_JSON = "user.json"

def read_pose(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    all_frame_points = data.get("all_frames_points", [])
    width = data.get("width", 0)
    height = data.get("height", 0)

    return width, height, all_frame_points

def calculate_score(width1, height1, all_frame_points1, width2, height2, all_frame_points2):
    return -1

async def get_scores_service(folder_id: str):
    root_path = os.path.join("data", folder_id)
    target_path = os.path.join(root_path, ORIGIN_JSON)
    user_path = os.path.join(root_path, USER_JSON)

    width1, height1, all_frame_points1 = read_pose(target_path)
    width2, height2, all_frame_points2 = read_pose(user_path)

    # 사용자 동영상과 원본 동영상 Score 계산
    score = calculate_score(width1, height1, all_frame_points1, width2, height2, all_frame_points2)
    
    return JSONResponse(content={"score": score}, status_code=200)