import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import video, pose, score, feedback

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    yield
    with open("app.log", mode="a") as log:
        log.write("Application shutdown\n")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(pose.router, prefix="/pose", tags=["Pose"])
app.include_router(score.router, prefix="/score", tags=["Score"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
