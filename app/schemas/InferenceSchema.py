from fastapi import File
from pydantic import BaseModel, Field, HttpUrl

class InferRequest(BaseModel):
    rail_mp4: str = Field(..., description="Rail MP4 URL")
    insulator_mp4: str = Field(..., description="Insulator MP4 URL")
    nest_mp4: str = Field(..., description="Nest MP4 URL")
    conf: float = Field(description="Confidence threshold", ge=0.0, le=1.0, default=0.25)
    iou: float = Field(description="IoU threshold", ge=0.1, le=1.0, default=0.7)
    stride: int = Field(description="Stride", ge=1, default=5)
