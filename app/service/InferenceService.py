from dataclasses import dataclass, asdict
from typing import Optional
import json

from .infer_rail import InferConfig as RailCfg, infer_mp4 as infer_rail
from .infer_insulator import InferConfig as InsCfg, infer_mp4 as infer_ins
from .infer_nest import InferConfig as NestCfg, infer_mp4 as infer_nest
from pathlib import Path
from huggingface_hub import hf_hub_download
import httpx
import tempfile
import zipfile
import io
import os


@dataclass
class ModelPaths:
    weights_path: Path
    yaml_path: Path


@dataclass
class DownloadedFile:
    path: Path
    original_filename: str


@dataclass
class InferenceResult:
    """각 영상의 추론 결과 상태"""
    status: str  # "success", "failed", "skipped"
    error: Optional[str] = None
    frames_processed: int = 0
    defects_found: int = 0

class InferenceService:
    def __init__(self):
        self.modelRootPath = Path(__file__).resolve().parent.parent.parent / "model"

        rail_repo = os.getenv("HF_RAIL_REPO", "stcheesecake-gh/rail-detector")
        ins_repo  = os.getenv("HF_INS_REPO",  "stcheesecake-gh/insulator-detector")
        nest_repo = os.getenv("HF_NEST_REPO", "stcheesecake-gh/nest-detector")

        rail_best = self._hf_best_pt(rail_repo)
        ins_best  = self._hf_best_pt(ins_repo)
        nest_best = self._hf_best_pt(nest_repo)

        self.railModelPaths = ModelPaths(
            weights_path=rail_best,
            yaml_path=self.modelRootPath / "rail" / "Yolo-v8n-laf" / "rail_hs_ns.yaml",
        )
        self.insulatorModelPaths = ModelPaths(
            weights_path=ins_best,
            yaml_path=self.modelRootPath / "insulator" / "Yolo-v8n-laf" / "insulator_hs_ns.yaml",
        )
        self.nestModelPaths = ModelPaths(
            weights_path=nest_best,
            yaml_path=self.modelRootPath / "nest" / "nest_hs_ns.yaml",
        )

        self._validate_model_paths()

    def _hf_best_pt(self, repo_id: str) -> Path:
        """
        Hugging Face model repo에서 weights/best.pt를 내려받아
        로컬 캐시 경로(Path)를 반환한다.
        """
        cache_dir = os.getenv("HF_CACHE_DIR", str(Path(__file__).resolve().parent.parent.parent / ".hf_cache"))
        revision = os.getenv("HF_REVISION")  # optional: "main" / tag / commit hash

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="weights/best.pt",
            repo_type="model",
            cache_dir=cache_dir,
            revision=revision if revision else None,
        )
        return Path(local_path)



    def _validate_model_paths(self):
        for modelPaths in [self.railModelPaths, self.insulatorModelPaths, self.nestModelPaths]:
            if not modelPaths.weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {modelPaths.weights_path}")
            if not modelPaths.yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {modelPaths.yaml_path}")

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        url: str,
        dest_path: Path,
        timeout: float = 30.0
    ) -> DownloadedFile:
        try:
            async with client.stream("GET", url, timeout=timeout) as response:
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                return DownloadedFile(path=dest_path, original_filename=Path(url).name)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to download file: {e}")

    def _zip_files(self, z: zipfile.ZipFile, base: Path, prefix: str):
        """디렉토리 내용을 ZIP에 추가"""
        if not base.exists():
            return
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p, arcname=f"{prefix}/{p.relative_to(base).as_posix()}")

    def _count_results(self, out_dir: Path) -> tuple[int, int]:
        """결과 디렉토리에서 프레임 수와 결함 수 계산"""
        frames = 0
        defects = 0
        if not out_dir.exists():
            return frames, defects
        
        for json_file in out_dir.rglob("*.json"):
            frames += 1
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "detections" in data:
                        defects += len(data["detections"])
                    elif isinstance(data, list):
                        defects += len(data)
            except Exception:
                pass
        return frames, defects

    async def run_inference(
        self,
        rail_url: Optional[str],
        insulator_url: Optional[str],
        nest_url: Optional[str],
        conf: float,
        iou: float,
        stride: int,
    ) -> io.BytesIO:
        """
        URL에서 영상 다운로드 후 추론 수행 (각 영상은 선택적)
        
        - URL이 없으면 해당 영상은 skipped
        - 다운로드/추론 실패 시 해당 영상은 failed 처리하고 나머지 계속 진행
        - 결과 ZIP에 summary.json 포함
        
        Returns:
            io.BytesIO: 결과 이미지 + JSON + summary.json이 담긴 ZIP 파일 버퍼
        """
        # 각 영상의 결과 상태 초기화
        results: dict[str, InferenceResult] = {
            "rail": InferenceResult(status="skipped"),
            "insulator": InferenceResult(status="skipped"),
            "nest": InferenceResult(status="skipped"),
        }
        
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            
            # 출력 디렉토리 생성
            out_root = td / "out"
            out_rail = out_root / "rail"
            out_ins = out_root / "insulator"
            out_nest = out_root / "nest"
            
            out_rail.mkdir(parents=True, exist_ok=True)
            out_ins.mkdir(parents=True, exist_ok=True)
            out_nest.mkdir(parents=True, exist_ok=True)
            
            async with httpx.AsyncClient() as client:
                # ========== Rail 처리 ==========
                if rail_url:
                    try:
                        rail_file = await self._download_file(
                            client, rail_url, td / "rail.mp4"
                        )
                        cfg_rail = RailCfg(
                            weights_path=str(self.railModelPaths.weights_path),
                            data_yaml_path=str(self.railModelPaths.yaml_path),
                            out_dir=str(out_rail),
                            conf=conf,
                            iou=iou,
                            stride=stride,
                            save_all=False,
                            keep_classes=None,
                            source_stem=os.path.splitext(rail_file.original_filename)[0],
                        )
                        infer_rail(str(rail_file.path), cfg_rail)
                        
                        frames, defects = self._count_results(out_rail)
                        results["rail"] = InferenceResult(
                            status="success",
                            frames_processed=frames,
                            defects_found=defects,
                        )
                    except Exception as e:
                        results["rail"] = InferenceResult(
                            status="failed",
                            error=str(e),
                        )
                
                # ========== Insulator 처리 ==========
                if insulator_url:
                    try:
                        ins_file = await self._download_file(
                            client, insulator_url, td / "insulator.mp4"
                        )
                        cfg_ins = InsCfg(
                            weights_path=str(self.insulatorModelPaths.weights_path),
                            data_yaml_path=str(self.insulatorModelPaths.yaml_path),
                            out_dir=str(out_ins),
                            conf=conf,
                            iou=iou,
                            stride=stride,
                            save_all=False,
                            keep_classes=None,
                            source_stem=os.path.splitext(ins_file.original_filename)[0],
                        )
                        infer_ins(str(ins_file.path), cfg_ins)
                        
                        frames, defects = self._count_results(out_ins)
                        results["insulator"] = InferenceResult(
                            status="success",
                            frames_processed=frames,
                            defects_found=defects,
                        )
                    except Exception as e:
                        results["insulator"] = InferenceResult(
                            status="failed",
                            error=str(e),
                        )
                
                # ========== Nest 처리 ==========
                if nest_url:
                    try:
                        nest_file = await self._download_file(
                            client, nest_url, td / "nest.mp4"
                        )
                        cfg_nest = NestCfg(
                            weights_path=str(self.nestModelPaths.weights_path),
                            data_yaml_path=str(self.nestModelPaths.yaml_path),
                            out_dir=str(out_nest),
                            conf=conf,
                            iou=iou,
                            stride=stride,
                            save_all=False,
                            keep_classes={0},
                            source_stem=os.path.splitext(nest_file.original_filename)[0],
                        )
                        infer_nest(str(nest_file.path), cfg_nest)
                        
                        frames, defects = self._count_results(out_nest)
                        results["nest"] = InferenceResult(
                            status="success",
                            frames_processed=frames,
                            defects_found=defects,
                        )
                    except Exception as e:
                        results["nest"] = InferenceResult(
                            status="failed",
                            error=str(e),
                        )
            
            # summary.json 생성
            summary = {k: asdict(v) for k, v in results.items()}
            summary_path = out_root / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # 결과 ZIP 생성
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # 성공한 결과만 포함
                if results["rail"].status == "success":
                    self._zip_files(z, out_rail, "rail")
                if results["insulator"].status == "success":
                    self._zip_files(z, out_ins, "insulator")
                if results["nest"].status == "success":
                    self._zip_files(z, out_nest, "nest")
                # summary.json은 항상 포함
                z.write(summary_path, arcname="summary.json")
            buf.seek(0)
            
            return buf


# 싱글톤 인스턴스
inference_service = InferenceService()