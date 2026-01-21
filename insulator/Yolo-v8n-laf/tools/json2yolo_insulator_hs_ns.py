#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ====== 프로젝트 루트(이 파일 기준) ======
ROOT = Path(__file__).resolve().parents[1]

# ====== RAW 데이터 루트 결정: data_insulator_raw 있으면 사용, 없으면 env INSULATOR_RAW_ROOT 사용 ======
RAW_CAND = ROOT / "data_insulator_raw"
ENV_RAW = os.environ.get("INSULATOR_RAW_ROOT", "").strip()
if RAW_CAND.exists():
    RAW_ROOT = RAW_CAND
elif ENV_RAW:
    RAW_ROOT = Path(ENV_RAW).expanduser().resolve()
else:
    raise RuntimeError("RAW 데이터 루트가 없습니다. data_insulator_raw를 만들거나 INSULATOR_RAW_ROOT를 지정하세요.")

# ====== 전역 메타데이터(다클래스 정의) ======
META_PATH = ROOT / "metadata" / "catenary_metadata.json"

# ====== 출력 YOLO 데이터셋 ======
OUT_ROOT = ROOT / "datasets" / "insulator_hs_ns"
COPY_IMAGES = True  # 안전하게 복사

SPLITS = {
    "train": {
        "img_root": RAW_ROOT / "Training" / "01.원천데이터",
        "lab_root": RAW_ROOT / "Training" / "02.라벨링데이터",
        "img_prefix": "TS_",
        "lab_prefix": "TL_",
    },
    "val": {
        "img_root": RAW_ROOT / "Validation" / "01.원천데이터",
        "lab_root": RAW_ROOT / "Validation" / "02.라벨링데이터",
        "img_prefix": "VS_",
        "lab_prefix": "VL_",
    },
}

# ✅ 여기서 사용할 supercategory를 지정
USE_SUPERCATS = {"고속철도", "일반철도"}  # 필요하면 "도시철도/경전철" 추가

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def polygon_flat_to_bbox(poly_flat):
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
    def c01(v): return 0.0 if v < 0 else 1.0 if v > 1 else v
    return c01(xc), c01(yc), c01(ww), c01(hh)

def load_global_map():
    meta = read_json(META_PATH)
    cats = meta.get("categories", []) or []

    # supercategory 필터
    filtered = [c for c in cats if c.get("supercategory") in USE_SUPERCATS]

    # (중요) 안정적인 순서를 위해 id로 정렬
    filtered = sorted(filtered, key=lambda c: int(c.get("id", 10**9)))

    old2new = {}
    names = []

    # supercategory → prefix 매핑(원하는 표기대로)
    prefix_map = {
        "고속철도": "고속철도",
        "일반철도": "일반철도",
        # 필요하면 추가:
        # "도시철도": "도시철도",
        # "경전철": "경전철",
    }

    for new_id, c in enumerate(filtered):
        old_id = int(c["id"])
        base_name = c.get("name_kor") or c.get("name") or str(old_id)

        supercat = c.get("supercategory", "")
        prefix = prefix_map.get(supercat, supercat if supercat else "UNKNOWN")

        # ✅ 최종 표시 이름: "고속철도_전차선" / "일반철도_전차선"
        disp = f"{prefix}_{base_name}"

        old2new[old_id] = new_id
        names.append(str(disp))

    if not names:
        raise RuntimeError(
            f"메타데이터에서 필터 결과가 비었습니다. META_PATH={META_PATH}, USE_SUPERCATS={USE_SUPERCATS}"
        )

    print("[DEBUG names sample]", names[:10])
    return old2new, names

def main():
    # out dirs
    for sp in SPLITS:
        ensure_dir(OUT_ROOT / "images" / sp)
        ensure_dir(OUT_ROOT / "labels" / sp)

    old2new, names = load_global_map()
    print(f"[GLOBAL] meta={META_PATH} | nc={len(names)}")

    stats = {sp: {"total": 0, "abnormal_labeled": 0, "normal_empty": 0, "copied": 0, "missing_img": 0} for sp in SPLITS}

    for sp, cfg in SPLITS.items():
        img_root, lab_root = cfg["img_root"], cfg["lab_root"]
        img_prefix, lab_prefix = cfg["img_prefix"], cfg["lab_prefix"]

        TARGETS = [
            ("일반철도", "정상"),
            ("일반철도", "이상"),
            ("고속철도", "정상"),
            ("고속철도", "이상"),
        ]

        for supercat, status in TARGETS:
            lab_dir = lab_root / f"{lab_prefix}{supercat}_{status}"
            img_dir = img_root / f"{img_prefix}{supercat}_{status}"

            if not lab_dir.exists() or not img_dir.exists():
                print(f"[SKIP] missing: {lab_dir} OR {img_dir}")
                continue

            json_files = sorted(lab_dir.glob("*.json"))
            print(f"[{sp}] {lab_dir.name}: json={len(json_files)}")

            for jp in json_files:
                j = read_json(jp)

                # image file name
                img_name = None
                imgs = j.get("image", [])
                if isinstance(imgs, list) and imgs:
                    img_name = imgs[0].get("file_name")
                if not img_name:
                    img_name = jp.stem + ".jpg"

                img_path = img_dir / img_name
                if not img_path.exists():
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

                meta = j.get("metadata", {}) or {}
                img_w = int(meta.get("width", 0) or 0)
                img_h = int(meta.get("height", 0) or 0)
                if img_w <= 0 or img_h <= 0:
                    continue

                # ✅ 핵심: 정상은 annotations를 아예 무시하고 "빈 txt"로 만든다.
                if status == "정상":
                    lines = []
                else:
                    lines = []
                    for ann in (j.get("annotations", []) or []):
                        old_cls = ann.get("category_id")
                        if old_cls is None:
                            continue
                        old_cls = int(old_cls)
                        if old_cls not in old2new:
                            continue

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

                out_img = OUT_ROOT / "images" / sp / img_path.name
                out_lab = OUT_ROOT / "labels" / sp / (img_path.stem + ".txt")

                if COPY_IMAGES and not out_img.exists():
                    shutil.copy2(img_path, out_img)
                    stats[sp]["copied"] += 1

                # ✅ 0바이트(완전 공란) 생성: join도 하지 말고 그냥 ""를 쓴다.
                if status == "정상":
                    out_lab.write_text("", encoding="utf-8")
                    stats[sp]["normal_empty"] += 1
                else:
                    out_lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    stats[sp]["abnormal_labeled"] += 1

                stats[sp]["total"] += 1

    # yaml save (상대경로 기준)
    y = OUT_ROOT / "insulator_hs_ns.yaml"
    yl = ["path: .", "train: images/train", "val: images/val", f"nc: {len(names)}", "names:"]
    yl += [f"  {i}: {n}" for i, n in enumerate(names)]
    y.write_text("\n".join(yl) + "\n", encoding="utf-8-sig")

    print("\n[DONE]")
    print(f"OUT_ROOT = {OUT_ROOT}")
    for sp in SPLITS:
        print(f"{sp}: total={stats[sp]['total']} abnormal_labeled={stats[sp]['abnormal_labeled']} normal_empty={stats[sp]['normal_empty']} copied={stats[sp]['copied']} missing_img={stats[sp]['missing_img']}")
    print(f"yaml: {y}")

if __name__ == "__main__":
    main()
