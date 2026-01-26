# tools/finetune/preprocess.py
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_det_list(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "detections" in j and j["detections"] is not None:
        return j["detections"]
    if "DETECTIONS" in j and j["DETECTIONS"] is not None:
        return j["DETECTIONS"]
    return []


def resolve_image_name(j: Dict[str, Any], json_path: Path) -> str:
    for k in ["image_file", "image", "file_name", "filename"]:
        if k in j and j[k]:
            return str(j[k])
    return json_path.stem + ".jpg"


def xyxy_to_yolo(b: List[float], W: int, H: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    # clamp
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return cx, cy, w, h


def load_cid_to_idx(class_map_path: Optional[Path]) -> Optional[Dict[int, int]]:
    if not class_map_path or not class_map_path.exists():
        return None
    cm = json.loads(class_map_path.read_text(encoding="utf-8"))
    m = cm.get("catid_to_idx")
    if not isinstance(m, dict):
        return None
    return {int(k): int(v) for k, v in m.items()}


def build_yolo_dataset(
    task: str,
    raw_task_dir: Path,      # data/<task>
    out_task_dir: Path,      # datasets/<task>
    class_map_path: Optional[Path] = None,  # nest만 사용 가능
) -> None:
    """
    input:
      data/<task>/origin/*.jpg
      data/<task>/json/*.json
    output:
      datasets/<task>/images/train/*.jpg
      datasets/<task>/labels/train/*.txt
    """
    img_in = raw_task_dir / "origin"
    js_in = raw_task_dir / "json"

    img_out = out_task_dir / "images/train"
    lb_out = out_task_dir / "labels/train"
    ensure_dir(img_out)
    ensure_dir(lb_out)

    cid_to_idx = load_cid_to_idx(class_map_path)

    json_files = sorted(js_in.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"[{task}] json not found: {js_in}")

    for jp in json_files:
        j = load_json(jp)

        img_name = resolve_image_name(j, jp)
        src_img = img_in / img_name
        if not src_img.exists():
            alt = img_in / (jp.stem + ".jpg")
            if alt.exists():
                src_img = alt
            else:
                raise FileNotFoundError(
                    f"[{task}] image not found for {jp.name}: {img_name} or {jp.stem+'.jpg'}"
                )

        dst_img = img_out / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        W, H = Image.open(dst_img).size
        dets = resolve_det_list(j)

        lines: List[str] = []
        for det in dets:
            cls = det.get("cls_id", det.get("class_id", det.get("cls")))
            if cls is None:
                continue
            cls = int(cls)

            # nest: cls가 1~4(cat id)로 올 수도 있어서 매핑 (있으면 적용)
            if cid_to_idx and cls in cid_to_idx:
                cls = cid_to_idx[cls]

            bbox = det.get("bbox_xyxy", det.get("BBOX_XYXY", det.get("bbox")))
            if bbox is None:
                continue

            cx, cy, w, h = xyxy_to_yolo([float(x) for x in bbox], W, H)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_txt = lb_out / (dst_img.stem + ".txt")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
