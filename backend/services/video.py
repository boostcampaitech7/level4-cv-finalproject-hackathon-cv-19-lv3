import os
from yt_dlp import YoutubeDL
from datetime import datetime
from fastapi.responses import JSONResponse, FileResponse
from config import logger
from constants import FilePaths

def download_video_service(request):
    try:
        # 원본 영상에 대한 ID 생성 및 폴더 생성
        folder_id = datetime.now().strftime("%y%m%d%H%M%S")
        root_path = os.path.join("data", folder_id)
        os.makedirs(root_path, exist_ok=True)
        video_path = os.path.join(root_path, FilePaths.ORIGIN_MP4.value)

        # 유튜브 영상 다운로드
        ydl_opts = {
            "outtmpl": video_path,
            "cookiesfrombrowser": ('firefox',),
            "verbose": True,
            "merge_output_format": "mp4"
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.url])

        logger.info(f"[{folder_id}] download YouTube video success")
        return JSONResponse(content={"folder_id": folder_id}, status_code=201)

    except Exception as e:
        logger.error(f"[{folder_id}] download YouTube video fail: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

def get_video_service(folder_id: str):
    try:
        video_path = os.path.join("data", folder_id, FilePaths.ORIGIN_MP4.value)

        logger.info(f"[{folder_id}] get downloaded video success")
        return FileResponse(video_path, media_type="video/mp4", filename=FilePaths.ORIGIN_MP4.value)

    except Exception as e:
        logger.error(f"[{folder_id}] get downloaded video fail: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
