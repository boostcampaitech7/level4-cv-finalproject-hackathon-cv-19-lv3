from pydantic import BaseModel, Field

class VideoRequest(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    folder_id: str = Field(..., alias="folderId")
    frame: str