from fastapi import APIRouter
from services.score import get_score_service

router = APIRouter()

@router.get("/{folder_id}")
async def get_scores(folder_id: str):
    return await get_score_service(folder_id)
