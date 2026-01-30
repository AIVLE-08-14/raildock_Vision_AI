# app/service/FinetuneService.py
import subprocess, os, json
from pathlib import Path
from datetime import datetime
import uuid

class FinetuneService:
    JOBS_DIR = Path("runs_finetune")  # ✅ 너 프로젝트에 이미 있는 폴더 사용

    @classmethod
    def start_job(cls, *,
                  tasks="rail,insulator,nest",
                  epochs=2, batch=2, imgsz=640, device="0",
                  hf_repo_rail="stcheesecake-gh/rail-detector",
                  hf_repo_insulator="stcheesecake-gh/insulator-detector",
                  hf_repo_nest="stcheesecake-gh/nest-detector",
                  hf_base_rail="weights/best.pt",
                  hf_base_insulator="weights/best.pt",
                  hf_base_nest="weights/best.pt"):
        cls.JOBS_DIR.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job_dir = cls.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        log_path = job_dir / "train.log"
        meta_path = job_dir / "meta.json"
        status_path = job_dir / "status.json"

        finetune_args = [
            "--tasks", tasks,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--device", str(device),
            "--hf-repo-rail", hf_repo_rail,
            "--hf-repo-insulator", hf_repo_insulator,
            "--hf-repo-nest", hf_repo_nest,
            "--hf-base-rail", hf_base_rail,
            "--hf-base-insulator", hf_base_insulator,
            "--hf-base-nest", hf_base_nest,
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
