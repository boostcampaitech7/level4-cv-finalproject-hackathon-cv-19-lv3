from fastapi import APIRouter, UploadFile, File, Form
from services.pose import extract_pose_from_video, extract_user_pose

router = APIRouter()

@router.get("/points/{folder_id}")
async def extract_origin_points(folder_id: str):
    return await extract_pose_from_video(folder_id)

@router.post("/user")
async def extract_user_points(folder_id: str = Form(...), video: UploadFile = File(...)):
    return await extract_user_pose(folder_id, video)