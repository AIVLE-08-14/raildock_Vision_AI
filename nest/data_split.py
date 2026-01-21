#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RailFOD23 COCO(train_val.json) -> stratified 80/10/10 split (seed=42)
+ export COCO jsons (train/val/test)
+ export YOLO labels (bbox) + dataset.yaml for Ultralytics YOLO

Directory (root_dir):
  data/
    Images/
    New_an/
      train_val.json

Outputs:
  data/New_an/splits_80_10_10_seed42/
    instances_train.json
    instances_val.json
    instances_test.json
    class_map.json
    split_summary.json
    yolo/
      labels/{train,val,test}/*.txt
      dataset.yaml

Stratification note:
- Detection is multi-label per image (an image can contain multiple classes).
- We use iterative multi-label stratification when available (recommended).
- Fallback: primary-label stratify (major label per image) if package is missing.
"""

import os
import json
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def coco_index(coco: Dict[str, Any]):
    # Build quick maps
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    img_by_id = {im["id"]: im for im in images}
    cat_by_id = {c["id"]: c for c in cats}

    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    # stable class order: sort by category id
    cat_ids_sorted = sorted([c["id"] for c in cats])
    catid_to_cidx = {cid: i for i, cid in enumerate(cat_ids_sorted)}
    cidx_to_catid = {i: cid for cid, i in catid_to_cidx.items()}
    cidx_to_name = {catid_to_cidx[c["id"]]: c.get("name", str(c["id"])) for c in cats}

    return img_by_id, anns_by_img, cat_by_id, catid_to_cidx, cidx_to_catid, cidx_to_name

def build_multilabel_matrix(image_ids: List[int],
                            anns_by_img: Dict[int, List[Dict[str, Any]]],
                            catid_to_cidx: Dict[int, int],
                            num_classes: int):
    """
    Y: list of multi-hot vectors (len=num_classes) for each image_id
    """
    Y = []
    for iid in image_ids:
        vec = [0] * num_classes
        for a in anns_by_img.get(iid, []):
            cid = a["category_id"]
            if cid in catid_to_cidx:
                vec[catid_to_cidx[cid]] = 1
        Y.append(vec)
    return Y

def iterative_stratified_split(image_ids: List[int], Y: List[List[int]],
                               train_ratio: float, val_ratio: float, test_ratio: float,
                               seed: int):
    """
    Prefer iterative multi-label stratification (best for detection).
    If not available, fallback to primary-label stratify.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    # Try iterative-stratification
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - train_ratio), random_state=seed
        )
        idxs = list(range(len(image_ids)))
        train_idx, tmp_idx = next(msss1.split(idxs, Y))

        tmp_ratio = val_ratio + test_ratio
        test_size2 = test_ratio / tmp_ratio  # within tmp
        Y_tmp = [Y[i] for i in tmp_idx]

        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size2, random_state=seed
        )
        tmp_local = list(range(len(tmp_idx)))
        val_local, test_local = next(msss2.split(tmp_local, Y_tmp))

        val_idx = [tmp_idx[i] for i in val_local]
        test_idx = [tmp_idx[i] for i in test_local]

        train_ids = [image_ids[i] for i in train_idx]
        val_ids = [image_ids[i] for i in val_idx]
        test_ids = [image_ids[i] for i in test_idx]
        return train_ids, val_ids, test_ids, "iterative_multilabel_stratify"

    except Exception as e:
        # Fallback: primary label stratify (majority class per image)
        try:
            from sklearn.model_selection import train_test_split
        except Exception:
            raise RuntimeError(
                "Fallback needs scikit-learn. Install either iterative-stratification "
                "or scikit-learn.\n"
                "pip install iterative-stratification scikit-learn"
            )

        # primary label: pick any present class; if multiple, choose the smallest index
        primary = []
        for vec in Y:
            ones = [i for i, v in enumerate(vec) if v == 1]
            primary.append(ones[0] if ones else -1)

        idxs = list(range(len(image_ids)))
        train_idx, tmp_idx = train_test_split(
            idxs, test_size=(1.0 - train_ratio), random_state=seed, stratify=primary
        )

        tmp_ratio = val_ratio + test_ratio
        test_size2 = test_ratio / tmp_ratio
        primary_tmp = [primary[i] for i in tmp_idx]

        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=test_size2, random_state=seed, stratify=primary_tmp
        )

        train_ids = [image_ids[i] for i in train_idx]
        val_ids = [image_ids[i] for i in val_idx]
        test_ids = [image_ids[i] for i in test_idx]
        return train_ids, val_ids, test_ids, "fallback_primarylabel_stratify"

def filter_coco_by_images(coco: Dict[str, Any], keep_image_ids: set) -> Dict[str, Any]:
    images = [im for im in coco["images"] if im["id"] in keep_image_ids]
    anns = [a for a in coco["annotations"] if a["image_id"] in keep_image_ids]
    return {
        "images": images,
        "annotations": anns,
        "categories": coco["categories"]
    }

def coco_to_yolo_labels(out_dir_labels: str,
                        image_ids: List[int],
                        img_by_id: Dict[int, Dict[str, Any]],
                        anns_by_img: Dict[int, List[Dict[str, Any]]],
                        catid_to_cidx: Dict[int, int]):
    """
    Writes one txt per image (same basename as image file, .txt)
    YOLO format: class x_center y_center w h  (normalized)
    """
    ensure_dir(out_dir_labels)

    num_written = 0
    num_skipped = 0

    for iid in image_ids:
        im = img_by_id[iid]
        w = float(im["width"])
        h = float(im["height"])
        file_name = im["file_name"]
        stem = os.path.splitext(os.path.basename(file_name))[0]
        out_txt = os.path.join(out_dir_labels, stem + ".txt")

        lines = []
        for a in anns_by_img.get(iid, []):
            bbox = a.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, bw, bh = bbox
            # guard invalid
            if bw <= 0 or bh <= 0:
                continue
            cid = a["category_id"]
            if cid not in catid_to_cidx:
                continue
            c = catid_to_cidx[cid]

            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h

            # clamp
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            lines.append(f"{c} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # YOLO는 empty 파일도 허용(= background), 하지만 여기선 명시적으로 생성
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if lines:
            num_written += 1
        else:
            num_skipped += 1

    return {"label_files_with_objects": num_written, "label_files_empty": num_skipped}

def compute_class_presence_stats(image_ids: List[int],
                                 anns_by_img: Dict[int, List[Dict[str, Any]]],
                                 catid_to_cidx: Dict[int, int],
                                 num_classes: int) -> List[int]:
    """
    count per class: number of images where the class appears at least once
    """
    counts = [0] * num_classes
    for iid in image_ids:
        present = set()
        for a in anns_by_img.get(iid, []):
            cid = a["category_id"]
            if cid in catid_to_cidx:
                present.add(catid_to_cidx[cid])
        for c in present:
            counts[c] += 1
    return counts

def main():
    set_seed(SEED)

    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    images_dir = os.path.join(data_dir, "Images")
    ann_dir = os.path.join(data_dir, "New_an")

    src_json = os.path.join(ann_dir, "train_val.json")
    if not os.path.exists(src_json):
        raise FileNotFoundError(f"Not found: {src_json}")

    coco = load_json(src_json)
    img_by_id, anns_by_img, cat_by_id, catid_to_cidx, cidx_to_catid, cidx_to_name = coco_index(coco)

    image_ids = [im["id"] for im in coco["images"]]
    num_classes = len(coco["categories"])

    # Build multilabel matrix for stratification
    Y = build_multilabel_matrix(image_ids, anns_by_img, catid_to_cidx, num_classes)

    # Split
    train_ids, val_ids, test_ids, strat_method = iterative_stratified_split(
        image_ids, Y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
    )

    # Output dir
    out_root = os.path.join(ann_dir, f"splits_80_10_10_seed{SEED}")
    ensure_dir(out_root)

    # Save split COCO jsons
    train_json = os.path.join(out_root, "instances_train.json")
    val_json = os.path.join(out_root, "instances_val.json")
    test_json = os.path.join(out_root, "instances_test.json")

    save_json(filter_coco_by_images(coco, set(train_ids)), train_json)
    save_json(filter_coco_by_images(coco, set(val_ids)), val_json)
    save_json(filter_coco_by_images(coco, set(test_ids)), test_json)

    # YOLO export
    yolo_root = os.path.join(out_root, "yolo")
    labels_root = os.path.join(yolo_root, "labels")
    ensure_dir(labels_root)

    stats_yolo = {}
    stats_yolo["train"] = coco_to_yolo_labels(os.path.join(labels_root, "train"), train_ids, img_by_id, anns_by_img, catid_to_cidx)
    stats_yolo["val"]   = coco_to_yolo_labels(os.path.join(labels_root, "val"),   val_ids,   img_by_id, anns_by_img, catid_to_cidx)
    stats_yolo["test"]  = coco_to_yolo_labels(os.path.join(labels_root, "test"),  test_ids,  img_by_id, anns_by_img, catid_to_cidx)

    # dataset.yaml for Ultralytics
    # IMPORTANT: we keep images in data/Images (no copy). Ultralytics can use absolute or relative paths.
    # We'll use relative paths from yolo_root:
    #   path: ../../..
    #   train: data/Images
    # and use labels path convention by setting "labels" folder under yolo/labels
    #
    # Ultralytics expects:
    #   path: <base>
    #   train: <images dir>
    #   val: <images dir>
    # But it also expects labels in <base>/<labels> with same relative structure.
    # To avoid moving images, we will write a "dataset.yaml" that uses:
    #   path: <root_dir>
    #   train: data/Images
    #   val: data/Images
    # and we will set "labels" by using Ultralytics 'data' loader option? (not supported directly)
    #
    # So we provide two options:
    # (A) Recommended: create symlinks yolo/images/{train,val,test} -> actual image files, matching label splits.
    # (B) Or use COCO json directly in other frameworks.
    #
    # We'll generate symlink folders if possible (Windows requires admin / dev mode).
    images_split_root = os.path.join(yolo_root, "images")
    ensure_dir(images_split_root)
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = os.path.join(images_split_root, split_name)
        ensure_dir(split_dir)
        # create symlinks if possible; else do nothing and just document it.
        linked = 0
        for iid in ids:
            im = img_by_id[iid]
            src_img = os.path.join(images_dir, im["file_name"])
            dst_img = os.path.join(split_dir, os.path.basename(im["file_name"]))
            if not os.path.exists(src_img):
                continue
            if os.path.exists(dst_img):
                continue
            try:
                os.symlink(src_img, dst_img)
                linked += 1
            except Exception:
                # if symlink fails, skip quietly
                pass
        stats_yolo[split_name]["image_symlinks_created"] = linked

    # Now write dataset.yaml that points to yolo/images/{train,val,test}
    names = [cidx_to_name[i] for i in range(num_classes)]
    dataset_yaml = {
        "path": os.path.abspath(yolo_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": names
    }
    with open(os.path.join(yolo_root, "dataset.yaml"), "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    # class map
    class_map = {
        "seed": SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "stratify_method": strat_method,
        "category_id_sorted": [cidx_to_catid[i] for i in range(num_classes)],
        "idx_to_name": {str(i): cidx_to_name[i] for i in range(num_classes)},
        "catid_to_idx": {str(cid): int(catid_to_cidx[cid]) for cid in catid_to_cidx}
    }
    save_json(class_map, os.path.join(out_root, "class_map.json"))

    # summary stats
    train_presence = compute_class_presence_stats(train_ids, anns_by_img, catid_to_cidx, num_classes)
    val_presence   = compute_class_presence_stats(val_ids,   anns_by_img, catid_to_cidx, num_classes)
    test_presence  = compute_class_presence_stats(test_ids,  anns_by_img, catid_to_cidx, num_classes)

    summary = {
        "seed": SEED,
        "stratify_method": strat_method,
        "counts": {"train_images": len(train_ids), "val_images": len(val_ids), "test_images": len(test_ids)},
        "class_presence_num_images": {
            "names": [cidx_to_name[i] for i in range(num_classes)],
            "train": train_presence,
            "val": val_presence,
            "test": test_presence
        },
        "yolo_export": stats_yolo
    }
    save_json(summary, os.path.join(out_root, "split_summary.json"))

    print("[DONE]")
    print(f"- source: {src_json}")
    print(f"- out:    {out_root}")
    print(f"- method: {strat_method}")
    print(f"- train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"- yolo dataset: {os.path.join(yolo_root, 'dataset.yaml')}")

if __name__ == "__main__":
    main()
