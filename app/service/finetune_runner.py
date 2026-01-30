# app/service/finetune_runner.py
import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

def _write_json(p: Path, obj: dict):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args, remaining = parser.parse_known_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    status_path = job_dir / "status.json"
    error_path = job_dir / "error.txt"

    _write_json(status_path, {"state": "running", "started_at": datetime.now().isoformat()})

    try:
        # finetune.py가 argparse로 sys.argv를 읽으므로 그대로 넘겨줌
        from tools.finetune import finetune as finetune_module

        # remaining이 ["--tasks","..."] 형태로 들어오게 만들 예정
        sys.argv = ["tools.finetune.finetune"] + remaining
        finetune_module.main()  # ✅ 여기서 학습 끝나고 HF 업로드까지 수행됨

        _write_json(status_path, {"state": "success", "finished_at": datetime.now().isoformat()})
        if error_path.exists():
            error_path.unlink()

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb, encoding="utf-8", errors="ignore")
        _write_json(status_path, {"state": "failed", "finished_at": datetime.now().isoformat()})
        raise

if __name__ == "__main__":
    main()
