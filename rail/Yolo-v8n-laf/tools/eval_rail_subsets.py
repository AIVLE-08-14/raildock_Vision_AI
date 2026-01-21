#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
import yaml

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_list(path_txt: Path, paths):
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    with path_txt.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")

def make_temp_yaml(base_dir: Path, src_yaml: dict, val_list_txt: Path, out_yaml: Path):
    # src_yaml의 nc/names 유지, val만 리스트(txt)로 교체
    tmp = {}
    tmp["path"] = str(base_dir)  # 절대/상대 상관없음
    tmp["train"] = src_yaml.get("train", "images/train")
    tmp["val"] = str(val_list_txt)
    tmp["nc"] = int(src_yaml["nc"])
    tmp["names"] = src_yaml["names"]
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(tmp, f, allow_unicode=True, sort_keys=False)

def metrics_row(tag: str, m):
    # m: DetMetrics
    return {
        "subset": tag,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="best.pt 경로")
    ap.add_argument("--data", type=str, required=True, help="rail_hs_ns.yaml 경로")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--outdir", type=str, default="", help="결과 저장 폴더(미지정 시 weights 폴더)")
    args = ap.parse_args()

    weights = Path(args.weights)
    data_yaml = Path(args.data)

    src = load_yaml(data_yaml)
    # YAML의 path: . 이라면 "yaml 파일 위치"를 기준으로 해석하는 게 안전
    data_root = data_yaml.parent  # datasets/rail_hs_ns
    val_dir = (data_root / src["val"]).resolve() if isinstance(src["val"], str) and "images" in src["val"] else None

    # 현재 네 YAML은 val: images/val 형태라서 이렇게 잡히는 게 정상
    if val_dir is None or not val_dir.exists():
        # fallback: datasets/rail_hs_ns/images/val
        val_dir = (data_yaml.parent / "images" / "val").resolve()

    # val 이미지 수집
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs = sorted([p for p in val_dir.rglob("*") if p.suffix.lower() in exts])

    # 파일명 prefix로 고속/일반 분리 (현재 네 파일명이 '고속철도_...' / '일반철도_...' 형태라 OK)
    hs_imgs = [p for p in all_imgs if p.stem.startswith("고속철도_")]
    nr_imgs = [p for p in all_imgs if p.stem.startswith("일반철도_")]

    # 출력 폴더
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = weights.parent.parent / "rail_metrics_custom"  # runs/detect/.../rail_metrics_custom
    outdir.mkdir(parents=True, exist_ok=True)

    # subset별 리스트 파일 + 임시 yaml 생성
    list_all = outdir / "val_all.txt"
    list_hs  = outdir / "val_highspeed.txt"
    list_nr  = outdir / "val_normal.txt"
    write_list(list_all, all_imgs)
    write_list(list_hs, hs_imgs)
    write_list(list_nr, nr_imgs)

    yaml_all = outdir / "data_all.yaml"
    yaml_hs  = outdir / "data_highspeed.yaml"
    yaml_nr  = outdir / "data_normal.yaml"
    make_temp_yaml(data_root.resolve(), src, list_all, yaml_all)
    make_temp_yaml(data_root.resolve(), src, list_hs,  yaml_hs)
    make_temp_yaml(data_root.resolve(), src, list_nr,  yaml_nr)

    model = YOLO(str(weights))

    # 1) subset별 “전체 지표” 계산
    rows_overall = []
    m_all = model.val(data=str(yaml_all), imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
    m_hs  = model.val(data=str(yaml_hs),  imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
    m_nr  = model.val(data=str(yaml_nr),  imgsz=args.imgsz, device=args.device, plots=False, verbose=False)

    rows_overall.append(metrics_row("ALL", m_all))
    rows_overall.append(metrics_row("HIGH_SPEED", m_hs))
    rows_overall.append(metrics_row("NORMAL", m_nr))

    # 콘솔 출력 (원하는 4개 지표)
    print("\n[METRICS SUMMARY]")
    for r in rows_overall:
        print(f"{r['subset']:>11} | P={r['precision']:.4f} R={r['recall']:.4f} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50-95']:.4f}")

    # 2) class별 지표: DetMetrics.summary() 사용
    # summary(): [{'class':0,'name':...,'precision':..,'recall':..,'map50':..,'map':..}, ...]
    def per_class_rows(tag, m):
        out = []
        for d in m.summary():
            out.append({
                "subset": tag,
                "class_id": d.get("class", ""),
                "class_name": d.get("name", ""),
                "precision": d.get("precision", ""),
                "recall": d.get("recall", ""),
                "mAP50": d.get("map50", ""),
                "mAP50-95": d.get("map", ""),
            })
        return out

    rows_class = []
    rows_class += per_class_rows("ALL", m_all)
    rows_class += per_class_rows("HIGH_SPEED", m_hs)
    rows_class += per_class_rows("NORMAL", m_nr)

    # 3) CSV 저장
    csv_overall = outdir / "metrics_overall.csv"
    csv_class   = outdir / "metrics_per_class.csv"

    with csv_overall.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "precision", "recall", "mAP50", "mAP50-95"])
        w.writeheader()
        w.writerows(rows_overall)

    with csv_class.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95"])
        w.writeheader()
        w.writerows(rows_class)

    print(f"\n[SAVED]\n- {csv_overall}\n- {csv_class}\n- {outdir}")

if __name__ == "__main__":
    main()
