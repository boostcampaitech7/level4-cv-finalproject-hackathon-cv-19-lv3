from fastapi import APIRouter
from models.model import FeedbackRequest
from services.feedback import get_frame_feedback_service

router = APIRouter()

@router.post("/")
async def get_frame_feedback(request: FeedbackRequest):
    return await get_frame_feedback_service(request)