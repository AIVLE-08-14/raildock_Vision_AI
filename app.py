import os
import io
import shutil
import zipfile
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse

from infer_rail import InferConfig as RailCfg, infer_mp4 as infer_rail
from infer_insulator import InferConfig as InsCfg, infer_mp4 as infer_ins
from infer_nest import InferConfig as NestCfg, infer_mp4 as infer_nest

app = FastAPI(title="Korail Multi Inference API", version="1.0.0")
ROOT = Path(__file__).resolve().parent

# ✅ 각 모델 폴더(실제 weights/yaml이 있는 곳)
RAIL_DIR = ROOT / "rail" / "Yolo-v8n-laf"
INS_DIR  = ROOT / "insulator" / "Yolo-v8n-laf"

# ✅ weights / yaml 경로를 baseline 기준으로 정확히 지정
RAIL_WEIGHTS = str(RAIL_DIR / "best.pt")
RAIL_YAML    = str(RAIL_DIR / "rail_hs_ns.yaml")         # rail 폴더에 있는 실제 파일명
INS_WEIGHTS  = str(INS_DIR / "best.pt")
INS_YAML     = str(INS_DIR / "insulator_hs_ns.yaml")     # insulator 폴더에 있는 실제 파일명


NEST_DIR = ROOT / "nest"
NEST_WEIGHTS = str(NEST_DIR / "best.pt")
NEST_YAML    = str(NEST_DIR / "nest_hs_ns.yaml")


def _save(upload: UploadFile, path: Path):
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

def _zip_dir(z: zipfile.ZipFile, base: Path, prefix: str):
    if not base.exists():
        return
    for p in base.rglob("*"):
        if p.is_file():
            z.write(p, arcname=f"{prefix}/{p.relative_to(base).as_posix()}")

@app.get("/health")
def health():
    return {"ok": True}

# ✅ 2개 MP4 동시에 받는 엔드포인트
@app.post("/predict_multi2")
def predict_multi2(
    rail_mp4: UploadFile = File(...),
    insulator_mp4: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.7, ge=0.0, le=1.0),
    stride: int = Query(5, ge=1),
):
    for f in (rail_mp4, insulator_mp4):
        if not f.filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only .mp4 is supported")

    # 파일 존재 체크(초기 셋업 실수 방지)
    for p in (RAIL_WEIGHTS, RAIL_YAML, INS_WEIGHTS, INS_YAML):
        if not os.path.exists(p):
            raise HTTPException(status_code=500, detail=f"Missing model file: {p}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ✅ OpenCV 한글 경로 이슈 방지: 내부 저장명은 ASCII 고정
        rail_path = td / "rail_input.mp4"
        ins_path  = td / "ins_input.mp4"
        _save(rail_mp4, rail_path)
        _save(insulator_mp4, ins_path)

        # 결과 폴더(모델별 분리)
        out_root = td / "out"
        out_rail = out_root / "rail"
        out_ins  = out_root / "insulator"
        out_rail.mkdir(parents=True, exist_ok=True)
        out_ins.mkdir(parents=True, exist_ok=True)

        # rail 추론
        cfg_r = RailCfg(
            weights_path=RAIL_WEIGHTS,
            data_yaml_path=RAIL_YAML,
            out_dir=str(out_rail),
            conf=conf, iou=iou, stride=stride,
            save_all=False,
            keep_classes=None,
            source_stem=os.path.splitext(rail_mp4.filename)[0],  # 원본 파일명 prefix
        )
        infer_rail(str(rail_path), cfg_r)

        # insulator 추론
        cfg_i = InsCfg(
            weights_path=INS_WEIGHTS,
            data_yaml_path=INS_YAML,
            out_dir=str(out_ins),
            conf=conf, iou=iou, stride=stride,
            save_all=False,
            keep_classes=None,
            source_stem=os.path.splitext(insulator_mp4.filename)[0],
        )
        infer_ins(str(ins_path), cfg_i)

        # zip 만들기: rail/... + insulator/...
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            _zip_dir(z, out_rail, "rail")
            _zip_dir(z, out_ins, "insulator")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="result.zip"'},
        )

@app.post("/predict_multi3")
def predict_multi3(
    rail_mp4: UploadFile = File(...),
    insulator_mp4: UploadFile = File(...),
    nest_mp4: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.7, ge=0.0, le=1.0),
    stride: int = Query(5, ge=1),
):
    for f in (rail_mp4, insulator_mp4, nest_mp4):
        if not f.filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only .mp4 is supported")

    # 파일 존재 체크
    for p in (RAIL_WEIGHTS, RAIL_YAML, INS_WEIGHTS, INS_YAML, NEST_WEIGHTS, NEST_YAML):
        if not os.path.exists(p):
            raise HTTPException(status_code=500, detail=f"Missing model file: {p}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        rail_path = td / "rail_input.mp4"
        ins_path  = td / "ins_input.mp4"
        nest_path = td / "nest_input.mp4"
        _save(rail_mp4, rail_path)
        _save(insulator_mp4, ins_path)
        _save(nest_mp4, nest_path)

        out_root = td / "out"
        out_rail = out_root / "rail"
        out_ins  = out_root / "insulator"
        out_nest = out_root / "nest"
        out_rail.mkdir(parents=True, exist_ok=True)
        out_ins.mkdir(parents=True, exist_ok=True)
        out_nest.mkdir(parents=True, exist_ok=True)

        cfg_r = RailCfg(
            weights_path=RAIL_WEIGHTS, data_yaml_path=RAIL_YAML, out_dir=str(out_rail),
            conf=conf, iou=iou, stride=stride, save_all=False, keep_classes=None,
            source_stem=os.path.splitext(rail_mp4.filename)[0],
        )
        infer_rail(str(rail_path), cfg_r)

        cfg_i = InsCfg(
            weights_path=INS_WEIGHTS, data_yaml_path=INS_YAML, out_dir=str(out_ins),
            conf=conf, iou=iou, stride=stride, save_all=False, keep_classes=None,
            source_stem=os.path.splitext(insulator_mp4.filename)[0],
        )
        infer_ins(str(ins_path), cfg_i)

        cfg_n = NestCfg(
            weights_path=NEST_WEIGHTS, data_yaml_path=NEST_YAML, out_dir=str(out_nest),
            conf=conf, iou=iou, stride=stride, save_all=False, keep_classes={0},
            source_stem=os.path.splitext(nest_mp4.filename)[0],
        )
        infer_nest(str(nest_path), cfg_n)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            _zip_dir(z, out_rail, "rail")
            _zip_dir(z, out_ins, "insulator")
            _zip_dir(z, out_nest, "nest")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="result.zip"'},
        )

