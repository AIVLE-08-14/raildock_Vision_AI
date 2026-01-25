# tools/finetune/hf_utils.py
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, HfApi


def require_hf_token() -> str:
    tok = os.getenv("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN 환경변수가 필요합니다. (write 권한 토큰)")
    return tok


def download_weight(repo_id: str, filename: str, revision: Optional[str] = None) -> str:
    # filename: repo 내부 경로 (예: weights/insulator/best.pt)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def upload_file(repo_id: str, local_path: Path, path_in_repo: str, commit_message: str) -> None:
    tok = require_hf_token()
    api = HfApi(token=tok)
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
