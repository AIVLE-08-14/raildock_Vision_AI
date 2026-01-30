from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import json
import zipfile, shutil, tempfile
import os
from app.service.FinetuneService import FinetuneService
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

class FinetuneRequest(BaseModel):
    tasks: str = "rail,insulator,nest"
    epochs: int = 2
    batch: int = 2
    imgsz: int = 640
    device: str = "0"
    hf_repo_rail: str
    hf_repo_insulator: str
    hf_repo_nest: str
    hf_base_rail: str = "weights/best.pt"
    hf_base_insulator: str = "weights/best.pt"
    hf_base_nest: str = "weights/best.pt"


@router.post("/finetune")
def finetune_start(req: FinetuneRequest):
    job_id, _ = FinetuneService.start_job(**req.model_dump())
    return {"job_id": job_id, "state": "running"}


@router.get("/finetune/{job_id}")
def finetune_status(job_id: str):
    job_dir = Path("runs_finetune") / job_id
    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(404, "job not found")
    status = json.loads(status_path.read_text(encoding="utf-8", errors="ignore"))
    return {"job_id": job_id, "status": status}


@router.get("/finetune/{job_id}/logs")
def finetune_logs(job_id: str, tail: int = 200):
    job_dir = Path("runs_finetune") / job_id
    log_path = job_dir / "train.log"
    if not log_path.exists():
        raise HTTPException(404, "log not found")
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {"job_id": job_id, "tail": tail, "lines": lines[-tail:]}


@router.post("/feedback")
async def upload_feedback(zip_file: UploadFile = File(...), overwrite: bool = False):
    """
    업로드 ZIP은 아래 구조를 포함해야 함:
      data/rail/origin/*.jpg
      data/rail/json/*.json
      data/insulator/origin/*.jpg
      data/insulator/json/*.json
      data/nest/origin/*.jpg
      data/nest/json/*.json

    서버는 이를 프로젝트 루트의 ./data/<task>/(origin|json)/ 로 병합(copy)함.
    """
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "zip_file must be a .zip")

    project_data = Path("data").resolve()
    tasks = ["rail", "insulator", "nest"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zip_path = td / "upload.zip"

        # 1) zip 저장
        with zip_path.open("wb") as f:
            shutil.copyfileobj(zip_file.file, f)

        # 2) zip 해제
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(td / "extracted")
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid zip file")

        extracted = td / "extracted"

        # 3) zip 내부의 data/ 경로 찾기 (최상위가 data/일 수도, 다른 폴더 밑에 data/가 있을 수도 있음)
        data_root = None
        for p in extracted.rglob("data"):
            if p.is_dir():
                data_root = p
                break
        if data_root is None:
            raise HTTPException(400, "ZIP must contain a 'data/' directory")

        # 4) 구조/매칭 검증 + 병합
        summary = {}
        for t in tasks:
            origin_dir = data_root / t / "origin"
            json_dir = data_root / t / "json"
            if not origin_dir.exists() or not json_dir.exists():
                # task가 아예 없을 수도 있으니, 여기서는 실패로 처리(원하면 skip로 바꿀 수 있음)
                raise HTTPException(400, f"Missing required folders: data/{t}/origin and data/{t}/json")

            jpgs = sorted(origin_dir.glob("*.jpg"))
            jsns = sorted(json_dir.glob("*.json"))

            jpg_bases = {p.stem for p in jpgs}
            json_bases = {p.stem for p in jsns}

            # 1:1 매칭 확인
            only_jpg = sorted(jpg_bases - json_bases)
            only_json = sorted(json_bases - jpg_bases)
            if only_jpg or only_json:
                raise HTTPException(
                    400,
                    f"[{t}] origin/json basename mismatch. "
                    f"only_jpg={only_jpg[:3]} only_json={only_json[:3]} (showing up to 3)"
                )

            # 병합 대상(프로젝트 실제 data)
            dst_origin = project_data / t / "origin"
            dst_json = project_data / t / "json"
            dst_origin.mkdir(parents=True, exist_ok=True)
            dst_json.mkdir(parents=True, exist_ok=True)

            copied = 0
            skipped = 0

            for base in sorted(jpg_bases):
                src_jpg = origin_dir / f"{base}.jpg"
                src_json = json_dir / f"{base}.json"
                dst_jpg = dst_origin / src_jpg.name
                dst_jsn = dst_json / src_json.name

                if (dst_jpg.exists() or dst_jsn.exists()) and not overwrite:
                    skipped += 1
                    continue

                shutil.copy2(src_jpg, dst_jpg)
                shutil.copy2(src_json, dst_jsn)
                copied += 1

            summary[t] = {"pairs": len(jpg_bases), "copied": copied, "skipped": skipped}

        return {"ok": True, "summary": summary, "overwrite": overwrite}