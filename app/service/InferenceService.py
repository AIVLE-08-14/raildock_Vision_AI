from dataclasses import dataclass

from .infer_rail import InferConfig as RailCfg, infer_mp4 as infer_rail
from .infer_insulator import InferConfig as InsCfg, infer_mp4 as infer_ins
from .infer_nest import InferConfig as NestCfg, infer_mp4 as infer_nest
from pathlib import Path

import httpx
import asyncio
import tempfile
import zipfile
import io
import os

@dataclass
class ModelPaths:
    weights_path: str
    yaml_path: str

@dataclass
class DownloadedFile:
    path: Path
    original_filename: str

class InferenceService:
    def __init__(self):
        self.modelRootPath = Path(__file__).resolve().parent.parent.parent / "model"
        self.railModelPaths = ModelPaths(
            weights_path=self.modelRootPath / "rail" / "Yolo-v8n-laf" / "best.pt",
            yaml_path=self.modelRootPath / "rail" / "Yolo-v8n-laf" / "rail_hs_ns.yaml",
        )
        self.insulatorModelPaths = ModelPaths(
            weights_path=self.modelRootPath / "insulator" / "Yolo-v8n-laf" / "best.pt",
            yaml_path=self.modelRootPath / "insulator" / "Yolo-v8n-laf" / "insulator_hs_ns.yaml",
        )
        self.nestModelPaths = ModelPaths(
            weights_path=self.modelRootPath / "nest" / "best.pt",
            yaml_path=self.modelRootPath / "nest" / "nest_hs_ns.yaml",
        )

        self._validate_model_paths()

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

    async def _download_all_files(
        self,
        rail_url: str,
        insulator_url: str,
        nest_url: str,
        temp_dir: Path,
        timeout: float = 30.0
    ) -> tuple[DownloadedFile, DownloadedFile, DownloadedFile]:
        async with httpx.AsyncClient() as client:
            tasks = [
                self._download_file(client, rail_url, temp_dir / "rail.mp4", timeout),
                self._download_file(client, insulator_url, temp_dir / "insulator.mp4", timeout),
                self._download_file(client, nest_url, temp_dir / "nest.mp4", timeout),
            ]
            downloaded_files = await asyncio.gather(*tasks)
            return tuple(downloaded_files)

    def _zip_files(self,
    z: zipfile.ZipFile,
    base: Path,
    prefix: str):
        if not base.exists():
            return
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p, arcname=f"{prefix}/{p.relative_to(base).as_posix()}")

    async def run_inference(
        self,
        rail_url: str,
        insulator_url: str,
        nest_url: str,
        conf: float,
        iou: float,
        stride: int,
    ) -> io.BytesIO:
        """
        S3 URL에서 영상 다운로드 후 추론 수행
        
        Returns:
            io.BytesIO: 결과 이미지 + JSON이 담긴 ZIP 파일 버퍼
        """
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            
            # 1. S3에서 파일 다운로드 (병렬)
            rail_file, ins_file, nest_file = await self._download_all_files(
                rail_url, insulator_url, nest_url, td
            )
            
            # 2. 출력 디렉토리 생성
            out_root = td / "out"
            out_rail = out_root / "rail"
            out_ins = out_root / "insulator"
            out_nest = out_root / "nest"
            
            out_rail.mkdir(parents=True, exist_ok=True)
            out_ins.mkdir(parents=True, exist_ok=True)
            out_nest.mkdir(parents=True, exist_ok=True)
            
            # 3. 각 모델 추론 실행 (동기 - YOLO는 CPU/GPU bound)
            # Rail 추론
            cfg_rail = RailCfg(
                weights_path=self.railModelPaths.weights_path,
                data_yaml_path=self.railModelPaths.yaml_path,
                out_dir=str(out_rail),
                conf=conf,
                iou=iou,
                stride=stride,
                save_all=False,
                keep_classes=None,
                source_stem=os.path.splitext(rail_file.original_filename)[0],
            )
            infer_rail(str(rail_file.path), cfg_rail)
            
            # Insulator 추론
            cfg_ins = InsCfg(
                weights_path=self.insulatorModelPaths.weights_path,
                data_yaml_path=self.insulatorModelPaths.yaml_path,
                out_dir=str(out_ins),
                conf=conf,
                iou=iou,
                stride=stride,
                save_all=False,
                keep_classes=None,
                source_stem=os.path.splitext(ins_file.original_filename)[0],
            )
            infer_ins(str(ins_file.path), cfg_ins)
            
            # Nest 추론
            cfg_nest = NestCfg(
                weights_path=self.nestModelPaths.weights_path,
                data_yaml_path=self.nestModelPaths.yaml_path,
                out_dir=str(out_nest),
                conf=conf,
                iou=iou,
                stride=stride,
                save_all=False,
                keep_classes={0},
                source_stem=os.path.splitext(nest_file.original_filename)[0],
            )
            infer_nest(str(nest_file.path), cfg_nest)
            
            # 4. 결과 ZIP 생성
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                self._zip_files(z, out_rail, "rail")
                self._zip_files(z, out_ins, "insulator")
                self._zip_files(z, out_nest, "nest")
            buf.seek(0)
            
            return buf


# 싱글톤 인스턴스
inference_service = InferenceService()