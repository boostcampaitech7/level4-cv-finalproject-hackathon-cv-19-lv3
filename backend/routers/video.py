from fastapi import APIRouter
from models.model import VideoRequest
from services.video import download_video_service, get_video_service

router = APIRouter()

@router.post("/url")
def download_video(request: VideoRequest):
    return download_video_service(request)

@router.get("/{folder_id}")
def get_video(folder_id: str):
    return get_video_service(folder_id)
