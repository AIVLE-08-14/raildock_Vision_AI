#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# 프로젝트 루트(이 파일 기준)
# =========================
LAF_ROOT = Path(__file__).resolve().parents[1]

# =========================
# RAW 데이터 루트 결정 규칙
# 1) LAF_ROOT/data_rail_raw 가 있으면 그걸 사용 (추천: 심볼릭링크/폴더복사)
# 2) 없으면 환경변수 RAIL_RAW_ROOT 사용
# =========================
RAW_CAND = LAF_ROOT / "data_rail_raw"
ENV_RAW = os.environ.get("RAIL_RAW_ROOT", "").strip()

if RAW_CAND.exists():
    SRC_ROOT = RAW_CAND
elif ENV_RAW:
    SRC_ROOT = Path(ENV_RAW).expanduser().resolve()
else:
    raise RuntimeError(
        "RAW 데이터 루트를 찾을 수 없습니다.\n"
        "해결 방법:\n"
        "  A) Yolo-v8n-laf/data_rail_raw 를 원본 데이터 폴더로 연결(심볼릭링크/복사)\n"
        "  B) 환경변수 RAIL_RAW_ROOT 에 원본 데이터 경로 지정\n"
    )

# =========================
# 출력 YOLO 데이터셋 (상대경로 고정)
# =========================
OUT_ROOT = LAF_ROOT / "datasets" / "rail_hs_ns"

# 고속/일반만 사용
USE_SUPERCATS = {"고속철도", "일반철도"}

SPLITS = {
    "train": {
        "img_root": SRC_ROOT / "Training" / "01.원천데이터",
        "lab_root": SRC_ROOT / "Training" / "02.라벨링데이터",
        "img_prefix": "TS_",
        "lab_prefix": "TL_",
    },
    "val": {
        "img_root": SRC_ROOT / "Validation" / "01.원천데이터",
        "lab_root": SRC_ROOT / "Validation" / "02.라벨링데이터",
        "img_prefix": "VS_",
        "lab_prefix": "VL_",
    },
}

# 안전: copy. (원하면 False로 바꾸고 링크 방식으로 커스텀 가능)
COPY_IMAGES = True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def polygon_flat_to_bbox(poly_flat):
    # poly_flat: [x1,y1,x2,y2,...]
    if not poly_flat or len(poly_flat) < 4:
        return None
    xs = poly_flat[0::2]
    ys = poly_flat[1::2]
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    if w <= 0 or h <= 0:
        return None
    return float(x_min), float(y_min), float(w), float(h)


def bbox_to_yolo(x, y, w, h, img_w, img_h):
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    ww = w / img_w
    hh = h / img_h

    def c01(v):
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    return c01(xc), c01(yc), c01(ww), c01(hh)


def build_class_map_from_categories(cats):
    """
    categories: [{"id":..,"name_kor":..,"supercategory":..}, ...]
    고속/일반 supercategory만 남기고 id를 0..K-1로 리매핑
    """
    filtered = [c for c in cats if c.get("supercategory") in USE_SUPERCATS]
    filtered = sorted(filtered, key=lambda c: int(c.get("id", 10**9)))

    old2new = {}
    names = []
    for new_id, c in enumerate(filtered):
        old_id = int(c["id"])
        nm = c.get("name_kor") or c.get("name") or str(old_id)
        old2new[old_id] = new_id
        names.append(str(nm))
    return old2new, names


def convert_one_json_to_yolo_lines(j):
    meta = j.get("metadata", {}) or {}
    img_w = int(meta.get("width", 0) or 0)
    img_h = int(meta.get("height", 0) or 0)

    cats = j.get("categories", []) or []
    old2new, names = build_class_map_from_categories(cats)

    lines = []
    for ann in (j.get("annotations", []) or []):
        old_cls = ann.get("category_id")
        if old_cls is None:
            continue
        old_cls = int(old_cls)
        if old_cls not in old2new:
            continue  # 고속/일반 아닌 클래스 제외

        bbox = ann.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4:
            x, y, w, h = map(float, bbox[:4])
            if w <= 0 or h <= 0:
                bb = polygon_flat_to_bbox(ann.get("polygon", []))
                if bb is None:
                    continue
                x, y, w, h = bb
        else:
            bb = polygon_flat_to_bbox(ann.get("polygon", []))
            if bb is None:
                continue
            x, y, w, h = bb

        yolo = bbox_to_yolo(x, y, w, h, img_w, img_h)
        if yolo is None:
            continue

        cls_new = old2new[old_cls]
        xc, yc, ww, hh = yolo
        lines.append(f"{cls_new} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    return lines, names

def load_global_map_from_metadata(meta_path: Path):
    meta = read_json(meta_path)
    cats = meta.get("categories", [])

    # 고속/일반만 필터
    filtered = [c for c in cats if c.get("supercategory") in USE_SUPERCATS]

    # 원본 id 기준 정렬 → 0..K-1로 리매핑
    filtered = sorted(filtered, key=lambda c: int(c.get("id", 10**9)))

    old2new = {}
    names = []
    for new_id, c in enumerate(filtered):
        old_id = int(c["id"])
        nm = c.get("name_kor") or c.get("name") or str(old_id)
        old2new[old_id] = new_id
        names.append(str(nm))
    return old2new, names

def convert_one_json_to_yolo_lines_with_global_map(j, global_old2new):
    meta = j.get("metadata", {}) or {}
    img_w = int(meta.get("width", 0) or 0)
    img_h = int(meta.get("height", 0) or 0)

    lines = []
    for ann in (j.get("annotations", []) or []):
        old_cls = ann.get("category_id")
        if old_cls is None:
            continue
        old_cls = int(old_cls)
        if old_cls not in global_old2new:
            continue  # 고속/일반 외 클래스 제외

        bbox = ann.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4:
            x, y, w, h = map(float, bbox[:4])
            if w <= 0 or h <= 0:
                bb = polygon_flat_to_bbox(ann.get("polygon", []))
                if bb is None:
                    continue
                x, y, w, h = bb
        else:
            bb = polygon_flat_to_bbox(ann.get("polygon", []))
            if bb is None:
                continue
            x, y, w, h = bb

        yolo = bbox_to_yolo(x, y, w, h, img_w, img_h)
        if yolo is None:
            continue

        cls_new = global_old2new[old_cls]
        xc, yc, ww, hh = yolo
        lines.append(f"{cls_new} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    return lines


def main():

    # 출력 폴더 생성

    meta_path = LAF_ROOT / "metadata" / "railway_metadata.json"
    global_old2new, global_names = load_global_map_from_metadata(meta_path)

    if len(global_names) == 0:
        raise RuntimeError("railway_metadata.json에서 고속/일반 클래스가 비었습니다.")

    print(f"[GLOBAL] meta={meta_path} | nc={len(global_names)}")

    for sp in SPLITS.keys():
        ensure_dir(OUT_ROOT / "images" / sp)
        ensure_dir(OUT_ROOT / "labels" / sp)

    stats = {"train": {"labeled": 0, "copied": 0, "missing_img": 0},
             "val":   {"labeled": 0, "copied": 0, "missing_img": 0}}

    for sp, cfg in SPLITS.items():
        img_root = cfg["img_root"]
        lab_root = cfg["lab_root"]
        img_prefix = cfg["img_prefix"]
        lab_prefix = cfg["lab_prefix"]

        for supercat in ["고속철도", "일반철도"]:
            for status in ["정상", "이상"]:
                img_dir = img_root / f"{img_prefix}{supercat}_{status}"
                lab_dir = lab_root / f"{lab_prefix}{supercat}_{status}"

                if not img_dir.exists() or not lab_dir.exists():
                    print(f"[SKIP] 폴더 없음: {img_dir} OR {lab_dir}")
                    continue

                json_files = sorted(lab_dir.glob("*.json"))
                print(f"[{sp}] {supercat}_{status}: json={len(json_files)}")

                for jp in json_files:
                    j = read_json(jp)

                    # 이미지 파일명: json의 image[0].file_name 우선
                    img_name = None
                    imgs = j.get("image", [])
                    if isinstance(imgs, list) and imgs:
                        img_name = imgs[0].get("file_name")

                    if not img_name:
                        img_name = jp.stem + ".jpg"

                    img_path = img_dir / img_name
                    if not img_path.exists():
                        # 확장자 다를 때 대비 (stem으로 재탐색)
                        stem = Path(img_name).stem
                        found = None
                        for ext in IMG_EXTS:
                            cand = img_dir / (stem + ext)
                            if cand.exists():
                                found = cand
                                break
                        if found is None:
                            stats[sp]["missing_img"] += 1
                            continue
                        img_path = found

                    lines = convert_one_json_to_yolo_lines_with_global_map(j, global_old2new)


                    out_img = OUT_ROOT / "images" / sp / img_path.name
                    out_lab = OUT_ROOT / "labels" / sp / (img_path.stem + ".txt")

                    if COPY_IMAGES and not out_img.exists():
                        shutil.copy2(img_path, out_img)
                        stats[sp]["copied"] += 1

                    # 객체 없으면 빈 txt 생성
                    out_lab.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
                    stats[sp]["labeled"] += 1



    yaml_path = OUT_ROOT / "rail_hs_ns.yaml"
    yaml_lines = []
    yaml_lines.append("path: .")  # yaml 파일 위치(=OUT_ROOT)를 기준으로 상대경로
    yaml_lines.append("train: images/train")
    yaml_lines.append("val: images/val")
    yaml_lines.append(f"nc: {len(global_names)}")
    yaml_lines.append("names:")
    for i, n in enumerate(global_names):
        yaml_lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("\n[DONE]")
    print(f"LAF_ROOT = {LAF_ROOT}")
    print(f"RAW_ROOT = {SRC_ROOT}")
    print(f"OUT_ROOT = {OUT_ROOT}")
    print(f"train: labeled={stats['train']['labeled']} copied={stats['train']['copied']} missing_img={stats['train']['missing_img']}")
    print(f"val  : labeled={stats['val']['labeled']}   copied={stats['val']['copied']}   missing_img={stats['val']['missing_img']}")
    print(f"yaml : {yaml_path}")


if __name__ == "__main__":
    main()
