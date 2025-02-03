from fastapi import APIRouter
from services.score import get_scores_service

router = APIRouter()

@router.get("/{folder_id}")
async def get_scores(folder_id: str):
    return await get_scores_service(folder_id)
