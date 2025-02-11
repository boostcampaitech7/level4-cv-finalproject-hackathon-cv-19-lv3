from fastapi import APIRouter
from models.model import FeedbackRequest
from services.feedback import get_frame_feedback_service, clear_cache_and_files

router = APIRouter()

@router.post("/")
async def get_frame_feedback(request: FeedbackRequest):
    return await get_frame_feedback_service(request)

@router.delete("/{folder_id}")
async def end_cycle_service(folder_id: str):
    return await clear_cache_and_files(folder_id)
