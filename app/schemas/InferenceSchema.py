from typing import Optional
from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    """추론 요청 스키마 - 모든 영상은 선택적"""
    rail_mp4: Optional[str] = Field(None, description="Rail MP4 URL")
    insulator_mp4: Optional[str] = Field(None, description="Insulator MP4 URL")
    nest_mp4: Optional[str] = Field(None, description="Nest MP4 URL")
    conf: float = Field(0.25, description="Confidence threshold", ge=0.0, le=1.0)
    iou: float = Field(0.7, description="IoU threshold", ge=0.1, le=1.0)
    stride: int = Field(5, description="Stride", ge=1)
