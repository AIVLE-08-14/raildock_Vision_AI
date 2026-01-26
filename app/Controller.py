from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .schemas.InferenceSchema import InferRequest
from .service.InferenceService import inference_service

router = APIRouter()


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/infer")
async def inference(request: InferRequest):
    """
    영상 추론 엔드포인트
    
    - 각 영상(rail, insulator, nest)은 선택적
    - 다운로드/추론 실패 시 해당 영상만 failed 처리되고 나머지는 계속 진행
    - 결과 ZIP에 summary.json 포함 (각 영상의 처리 상태)
    """
    try:
        result = await inference_service.run_inference(
            rail_url=request.rail_mp4,
            insulator_url=request.insulator_mp4,
            nest_url=request.nest_mp4,
            conf=request.conf,
            iou=request.iou,
            stride=request.stride,
        )
        return StreamingResponse(
            result,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="result.zip"'},
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )