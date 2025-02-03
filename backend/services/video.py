import os
from datetime import datetime
from fastapi.responses import JSONResponse, FileResponse
from yt_dlp import YoutubeDL

ORIGIN_VIDEO = "video.mp4"

def download_video_service(request):
    try:
        # 새로운 id 생성
        folder_id = datetime.now().strftime("%y%m%d%H%M%S")
        root_path = os.path.join("data", folder_id)
        os.makedirs(root_path, exist_ok=True)

        video_path = os.path.join(root_path, ORIGIN_VIDEO)

        # url로부터 YouTube video 추출
        ydl_opts = {
            "outtmpl": video_path,
            "cookiesfrombrowser": ('firefox',),
            "verbose": True,
            "merge_output_format": "mp4"
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.url])

        return JSONResponse(content={"folder_id": folder_id}, status_code=201)

    except Exception:
        return JSONResponse(content={"error": "Failed YouTube video extract"}, status_code=400)

def get_video_service(folder_id: str):
    video_path = os.path.join("data", folder_id, ORIGIN_VIDEO)
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=ORIGIN_VIDEO)
    return JSONResponse(content={"error": "File not found"}, status_code=400)