from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from .schemas.InferenceSchema import InferRequest
from .service.InferenceService import inference_service
import httpx
router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True}

@router.post("/infer")
async def inference(request: InferRequest):
    try:
        result = await inference_service.run_inference(
            request.rail_mp4,
            request.insulator_mp4,
            request.nest_mp4,
            request.conf,
            request.iou,
            request.stride,
        )
        return StreamingResponse(
            result,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="result.zip"'},
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download file: {e}"
            )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"File not found: {str(e)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
            )