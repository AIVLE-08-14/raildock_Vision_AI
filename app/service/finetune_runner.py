# app/service/finetune_runner.py
import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
import shutil  # 추가

def _write_json(p: Path, obj: dict):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _safe_rmtree(path: Path):
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    except Exception as e:
        print(f"[WARN] cleanup failed: {path} ({e})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args, remaining = parser.parse_known_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    status_path = job_dir / "status.json"
    error_path = job_dir / "error.txt"

    _write_json(status_path, {"state": "running", "started_at": datetime.now().isoformat()})

    success = False  # ✅ 추가

    try:
        from tools.finetune import finetune as finetune_module

        sys.argv = ["tools.finetune.finetune"] + remaining
        finetune_module.main()

        success = True  # ✅ 성공 표시

        _write_json(status_path, {"state": "success", "finished_at": datetime.now().isoformat()})
        if error_path.exists():
            error_path.unlink()

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb, encoding="utf-8", errors="ignore")
        _write_json(status_path, {"state": "failed", "finished_at": datetime.now().isoformat()})
        raise

    finally:
        if success:
            project_root = Path.cwd()

            # ✅ success일 때만 삭제
            _safe_rmtree(project_root / "data")
            _safe_rmtree(project_root / "datasets")

if __name__ == "__main__":
    main()
