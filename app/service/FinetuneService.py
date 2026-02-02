# app/service/FinetuneService.py
import subprocess, os, json
from pathlib import Path
from datetime import datetime
import uuid
from typing import Dict, Any

class FinetuneService:
    JOBS_DIR = Path("runs_finetune")
    CONFIG_PATH = Path("config/finetune_config.json")  # ✅ 서버 재시작해도 유지되게 파일로 저장

    # ✅ 고정 디폴트(요구사항: API로 변경 불가)
    DEFAULTS = {
        "tasks": "rail,insulator,nest",
        "device": "0",
        "hf_repo_rail": "stcheesecake-gh/rail-detector",
        "hf_repo_insulator": "stcheesecake-gh/insulator-detector",
        "hf_repo_nest": "stcheesecake-gh/nest-detector",
        "hf_base_rail": "weights/best.pt",
        "hf_base_insulator": "weights/best.pt",
        "hf_base_nest": "weights/best.pt",
    }

    # ✅ 변경 가능한 파라미터 기본값
    DEFAULT_TRAINING = {
        "epochs": 2,
        "batch": 2,
        "imgsz": 640,
    }

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        현재 epochs/batch/imgsz 설정을 반환.
        파일이 없으면 기본값 생성해서 저장 후 반환.
        """
        cls.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if cls.CONFIG_PATH.exists():
            try:
                cfg = json.loads(cls.CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        else:
            cfg = {}

        # 누락 키 채우기
        merged = dict(cls.DEFAULT_TRAINING)
        for k in merged.keys():
            if k in cfg and isinstance(cfg[k], int):
                merged[k] = cfg[k]

        # 파일이 없거나 깨졌으면 정상화해서 저장
        cls.CONFIG_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        return merged

    @classmethod
    def set_config(cls, *, epochs: int, batch: int, imgsz: int) -> Dict[str, Any]:
        """
        epochs/batch/imgsz만 갱신 (고정 파라미터는 건드리지 않음)
        """
        # 최소 검증(원하면 더 강화 가능)
        if epochs <= 0 or batch <= 0 or imgsz <= 0:
            raise ValueError("epochs/batch/imgsz must be positive integers")

        cfg = {"epochs": int(epochs), "batch": int(batch), "imgsz": int(imgsz)}
        cls.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return cfg

    @classmethod
    def start_job(cls, *, epochs: int, batch: int, imgsz: int):
        """
        (중요) 파인튜닝 job 시작.
        - tasks/device/hf_repo/hf_base는 DEFAULTS로 고정
        - epochs/batch/imgsz는 config에서 받은 값만 사용
        """
        cls.JOBS_DIR.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job_dir = cls.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        log_path = job_dir / "train.log"
        meta_path = job_dir / "meta.json"
        status_path = job_dir / "status.json"

        d = cls.DEFAULTS

        finetune_args = [
            "--tasks", d["tasks"],
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--device", d["device"],
            "--hf-repo-rail", d["hf_repo_rail"],
            "--hf-repo-insulator", d["hf_repo_insulator"],
            "--hf-repo-nest", d["hf_repo_nest"],
            "--hf-base-rail", d["hf_base_rail"],
            "--hf-base-insulator", d["hf_base_insulator"],
            "--hf-base-nest", d["hf_base_nest"],
        ]

        cmd = [
            "uv", "run", "python", "-m", "app.service.finetune_runner",
            "--job-dir", str(job_dir),
            *finetune_args
        ]

        meta_path.write_text(json.dumps({
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "cmd": cmd,
            "training": {"epochs": epochs, "batch": batch, "imgsz": imgsz},
            "fixed": d,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        status_path.write_text(json.dumps({"state": "queued"}, ensure_ascii=False, indent=2), encoding="utf-8")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        with open(log_path, "a", encoding="utf-8", errors="ignore") as f:
            subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
                creationflags=creationflags,
            )

        return job_id, job_dir

    @classmethod
    def get_active_job_id(cls):
        jobs_dir = cls.JOBS_DIR
        if not jobs_dir.exists():
            return None

        candidates = []
        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            meta_path = job_dir / "meta.json"
            created_at = ""
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
                    created_at = str(meta.get("created_at", ""))
                except Exception:
                    pass
            candidates.append((created_at, job_dir))

        # 최신 created_at 먼저
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, job_dir in candidates:
            status_path = job_dir / "status.json"
            if not status_path.exists():
                continue
            try:
                status = json.loads(status_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            state = str(status.get("state", "")).lower()
            if state in ("queued", "running"):
                return job_dir.name

        return None

    @classmethod
    def is_training_active(cls) -> bool:
        return cls.get_active_job_id() is not None