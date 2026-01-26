# tools/finetune/finetune.py
import argparse
from pathlib import Path
import yaml

from ultralytics import YOLO

from tools.finetune.preprocess import build_yolo_dataset
from tools.finetune.hf_utils import download_weight, upload_file

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # repo root

MODEL_YAML = {
    "rail": ROOT / "model/rail/Yolo-v8n-laf/rail_hs_ns.yaml",
    "insulator": ROOT / "model/insulator/Yolo-v8n-laf/insulator_hs_ns.yaml",
    "nest": ROOT / "model/nest/nest_hs_ns.yaml",
}


def patch_data_yaml(yaml_path: Path, dataset_root: Path) -> None:
    """
    ZIP 안 yaml에 절대경로 박혀있는 걸 깨끗하게 고침.
    val 없으면 train으로 지정해서 학습 안 깨지게 처리.
    """
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))
    data["path"] = str(dataset_root.as_posix())
    data["train"] = "images/train"
    data["val"] = "images/train"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="rail,insulator,nest")
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets-root", default="datasets")
    p.add_argument("--runs-root", default="runs_finetune")

    # train args
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--device", default="0")

    # HF
    p.add_argument("--hf-repo-rail", required=True)
    p.add_argument("--hf-repo-insulator", required=True)
    p.add_argument("--hf-repo-nest", required=True)

    p.add_argument("--hf-base-rail", required=True)       # repo 내부 경로 (예: weights/rail/best.pt)
    p.add_argument("--hf-base-insulator", required=True)
    p.add_argument("--hf-base-nest", required=True)

    p.add_argument("--hf-revision", default=None)         # main 등 공통
    args = p.parse_args()

    data_root = ROOT / args.data_root
    ds_root = ROOT / args.datasets_root
    runs_root = ROOT / args.runs_root

    class_map_path = ROOT / "class_map.json"  # 있으면 nest에 사용

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in task_list:
        raw_task_dir = data_root / task
        out_task_dir = ds_root / task

        # 1) 전처리: json -> yolo txt + datasets 구성
        build_yolo_dataset(
            task=task,
            raw_task_dir=raw_task_dir,
            out_task_dir=out_task_dir,
            class_map_path=(class_map_path if task == "nest" else None),
        )

        # 2) yaml 절대경로 패치
        yaml_path = MODEL_YAML[task]
        patch_data_yaml(yaml_path, out_task_dir)

        # 3) HF에서 base weight 다운로드
        if task == "rail":
            repo_id, base_name = args.hf_repo_rail, args.hf_base_rail
        elif task == "insulator":
            repo_id, base_name = args.hf_repo_insulator, args.hf_base_insulator
        else:
            repo_id, base_name = args.hf_repo_nest, args.hf_base_nest

        base_path = download_weight(repo_id=repo_id, filename=base_name, revision=args.hf_revision)

        # 4) finetune
        project_dir = runs_root / task
        model = YOLO(base_path)
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            device=args.device,
            project=str(project_dir),
            name="train",
            exist_ok=True,
        )

        best = project_dir / "train" / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"[{task}] best.pt not found: {best}")

        # 5) HF 업로드 (repo 내 저장 위치 통일 추천)
        upload_path = f"weights/best.pt"
        upload_file(
            repo_id=repo_id,
            local_path=best,
            path_in_repo=upload_path,
            commit_message=f"finetune: update {task} best.pt",
        )
        print(f"[{task}] ✅ uploaded: {repo_id}/weights/best.pt")


if __name__ == "__main__":
    main()
