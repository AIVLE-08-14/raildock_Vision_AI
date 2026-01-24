import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
from PIL import Image, ImageDraw, ImageFont
import cv2
import yaml
from ultralytics import YOLO
import re
import numpy as np

# 보기 좋은 팔레트(BGR 아님, PIL이 RGB를 쓰니 RGB로)
PALETTE = [
    (255,  80,  80),   # red-ish
    ( 80, 255,  80),   # green-ish
    ( 80, 160, 255),   # blue-ish
    (255, 200,  80),   # orange-ish
    (200,  80, 255),   # purple-ish
    ( 80, 255, 255),   # cyan-ish
    (255, 120, 200),   # pink-ish
    (180, 180, 255),   # lavender-ish
]

def color_for_cls(cls_id: int):
    return PALETTE[cls_id % len(PALETTE)]


def _norm_label(s: str) -> str:
    # 공백/언더바/대소문자 변형에 강하게 매칭하려고 정규화
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)      # 연속 공백 정리
    s = s.replace("_", " ")         # 언더바 → 공백으로 통일
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ✅ 너가 준 이상상세(status_detail_clean) 기준으로 "표시 이름" 변경
#    (기존 raw label이 공백/언더바 섞여 있어도 매칭되게 norm_key로 관리)
LABEL_REMAP = {
    _norm_label("고속철도 애자 류"):     "고속철도_애자 류_균열 파손",
    _norm_label("고속철도 전차선"):      "고속철도_전차선_마모 커버 탈락",
    _norm_label("고속철도 클램프 류"):   "고속철도_클램프 류_탈락",
    _norm_label("고속철도 프로텍터"):    "고속철도_프로텍터_파손",
    _norm_label("고속철도 행거"):        "고속철도_행거_이탈",

    _norm_label("일반철도 애자 류"):     "일반철도_애자 류_균열 파손",
    _norm_label("일반철도 전차선"):      "일반철도_전차선_마모 등",
    _norm_label("일반철도 클램프 류"):   "일반철도_클램프 류_탈락",
    _norm_label("일반철도 프로텍터"):    "일반철도_프로텍터_파손",
    _norm_label("일반철도 행거"):        "일반철도_행거_이탈",
}



@dataclass
class InferConfig:
    weights_path: str                  # best.pt 경로
    data_yaml_path: Optional[str] = None  # rail_hs_ns.yaml (선택, names 덮어쓰기용)

    out_dir: str = "outputs"
    conf: float = 0.25
    iou: float = 0.7

    stride: int = 5          # N프레임마다 1번 추론
    save_all: bool = False   # 탐지 없을 때도 저장할지 (False면 '이상' 프레임만 저장)
    keep_classes: Optional[Set[int]] = None
    # keep_classes 예: {0, 3} 처럼 특정 클래스만 이상으로 취급하고 싶을 때 사용
    # None이면 탐지 1개라도 있으면 이상 처리
    source_stem: Optional[str] = None   # ✅ 추가: 출력 파일명 prefix용(원본 mp4 이름)


def load_names_from_yaml(yaml_path: str) -> Optional[Dict[int, str]]:
    if not yaml_path or not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if not isinstance(names, dict):
        return None
    out = {}
    for k, v in names.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out if out else None


def _get_cls_name(names_obj, cls_id: int) -> str:
    # ultralytics result.names 는 보통 dict 형태
    if isinstance(names_obj, dict):
        return str(names_obj.get(cls_id, cls_id))
    if isinstance(names_obj, list) and 0 <= cls_id < len(names_obj):
        return str(names_obj[cls_id])
    return str(cls_id)


def _remap_name(raw_name: str) -> str:
    key = _norm_label(raw_name)
    return LABEL_REMAP.get(key, raw_name)

def split_label_to_fields(label: str):
    """
    label 예:
      - "고속철도_FAST clip_훼손"
      - "일반철도_볼트너트_너트 풀림"
      - (혹시 _가 없으면) "고속철도_FAST clip"
    반환: (rail_type, cls_name, detail)
    """
    s = str(label).strip()

    # 혹시 라벨에 공백으로만 되어있을 수도 있으니, 먼저 언더바 기준 분리
    parts = s.split("_")

    # 1) rail_type
    rail_type = parts[0].strip() if len(parts) >= 1 else ""

    # 2) cls_name
    cls_name = parts[1].strip() if len(parts) >= 2 else ""

    # 3) detail: 3번째 이후가 있으면 '_'로 다시 합쳐서 detail로
    detail = "_".join(parts[2:]).strip() if len(parts) >= 3 else ""

    return rail_type, cls_name, detail


def _get_korean_font(font_size: int = 28):
    # Linux(Docker) + Windows 둘 다 지원
    candidates = [
        # --- Linux (Docker) ---
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",

        # --- Windows ---
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunsl.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        r"C:\Windows\Fonts\batang.ttc",
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass

    # 폰트 못 찾으면 기본 폰트(한글은 깨질 수 있음)
    return ImageFont.load_default()


def draw_boxes_with_remap(frame, boxes, names_obj):
    """
    - 클래스별 색상 다르게
    - bbox 두께 크게
    - 라벨 폰트 크게 + 배경(반투명 느낌) + padding
    - 라벨이 프레임 밖으로 절대 나가지 않게 + 서로 최대한 안 겹치게
    """
    if boxes is None or len(boxes) == 0:
        return frame

    # OpenCV(BGR) -> PIL(RGB)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))  # 투명 오버레이
    draw = ImageDraw.Draw(overlay)

    font = _get_korean_font(font_size=30)
    bbox_thick = 4

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    W, H = pil.size
    placed = []  # 지금까지 배치된 라벨 박스들 (x0,y0,x1,y1)

    def _overlap(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    for i in range(len(clss)):
        cls_id = int(clss[i])
        raw = _get_cls_name(names_obj, cls_id)
        name = _remap_name(raw)
        conf = float(confs[i])

        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        # 클래스별 색
        r, g, b = color_for_cls(cls_id)

        # bbox
        for t in range(bbox_thick):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 255), width=1)

        # 라벨 텍스트
        text = f"{name} {conf:.2f}"

        # 텍스트 크기
        l, t, rr, bb = draw.textbbox((0, 0), text, font=font)
        tw, th = (rr - l), (bb - t)

        pad_x, pad_y = 10, 6
        label_w = tw + pad_x * 2
        label_h = th + pad_y * 2

        # -------------------------
        # 1) 초기 후보 위치 결정: 위 -> 아래 -> bbox 안쪽
        # -------------------------
        tx = x1

        ty_above = y1 - label_h - 2
        ty_below = y2 + 2

        if ty_above >= 0:
            ty = ty_above
        elif ty_below + label_h <= H:
            ty = ty_below
        else:
            # 위도 아래도 불가면 bbox 안쪽
            ty = max(0, min(y1 + 2, H - label_h - 1))

        # -------------------------
        # 2) 시작 위치 1차 클램프 (프레임 밖 금지)
        # -------------------------
        tx = max(0, min(tx, W - label_w - 1))
        ty = max(0, min(ty, H - label_h - 1))

        # -------------------------
        # 3) 겹치면 아래로 밀면서 찾기 (항상 프레임 내부 유지)
        # -------------------------
        max_tries = 30
        step = label_h + 2

        best = None
        for _ in range(max_tries):
            # 매 시도마다 프레임 안으로 강제
            tx = max(0, min(tx, W - label_w - 1))
            ty = max(0, min(ty, H - label_h - 1))

            x0, y0 = tx, ty
            x1b = min(tx + label_w, W - 1)
            y1b = min(ty + label_h, H - 1)
            cand = (x0, y0, x1b, y1b)

            hit = False
            for p in placed:
                if _overlap(cand, p):
                    hit = True
                    break

            if not hit:
                best = cand
                placed.append(cand)
                break

            # 겹치면 아래로 한 칸 내리기
            ty += step

            # 바닥에 닿으면: bbox 안쪽으로 강제(옛날처럼 위로 리셋하지 않음)
            if ty > H - label_h - 1:
                ty = max(0, min(y1 + 2, H - label_h - 1))

        # best를 못 찾으면 현재 위치로(그래도 프레임 밖은 아님)
        if best is None:
            x0, y0 = tx, ty
            x1b = min(tx + label_w, W - 1)
            y1b = min(ty + label_h, H - 1)
        else:
            x0, y0, x1b, y1b = best

        # 최종 안전 클램프
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1b = max(x0 + 1, min(x1b, W - 1))
        y1b = max(y0 + 1, min(y1b, H - 1))

        # 배경 (반투명)
        draw.rectangle([x0, y0, x1b, y1b], fill=(r, g, b, 110))

        # 텍스트
        text_x = x0 + pad_x
        text_y = y0 + pad_y
        try:
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
        except TypeError:
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                draw.text((text_x + dx, text_y + dy), text, font=font, fill=(0, 0, 0, 255))
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(pil, overlay).convert("RGB")
    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out_bgr





def infer_mp4(mp4_path: str, cfg: InferConfig) -> Dict[str, Any]:
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"mp4 not found: {mp4_path}")
    if not os.path.exists(cfg.weights_path):
        raise FileNotFoundError(f"weights not found: {cfg.weights_path}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    detect_dir = os.path.join(cfg.out_dir, "detect")
    origin_dir = os.path.join(cfg.out_dir, "origin")
    json_dir = os.path.join(cfg.out_dir, "json")
    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 모델 로드
    model = YOLO(cfg.weights_path)

    # (선택) yaml names로 덮어쓰기 — “yaml 바꾸면 추론도 같은 이름으로 나오게”
    names_map = load_names_from_yaml(cfg.data_yaml_path) if cfg.data_yaml_path else None

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    processed = 0
    saved = 0
    items: List[Dict[str, Any]] = []

    t0 = time.time()
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % cfg.stride != 0:
            frame_index += 1
            continue

        processed += 1
        timestamp_ms = (frame_index / fps * 1000.0) if fps > 0 else float(frame_index)

        # 추론
        results = model.predict(
            source=frame,
            conf=cfg.conf,
            iou=cfg.iou,
            verbose=False
        )
        r = results[0]

        boxes = r.boxes
        has_det = boxes is not None and len(boxes) > 0

        # keep_classes 필터 적용 (특정 클래스만 이상 취급)
        det_indices = []
        if has_det:
            cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            if cfg.keep_classes is None:
                det_indices = list(range(len(cls_ids)))
            else:
                det_indices = [i for i, c in enumerate(cls_ids) if c in cfg.keep_classes]
                has_det = len(det_indices) > 0

        # 저장 여부 결정
        if (not has_det) and (not cfg.save_all):
            frame_index += 1
            continue

        # annotated JPG 저장
        annotated = draw_boxes_with_remap(frame, r.boxes, r.names)
        # mp4_stem = os.path.splitext(os.path.basename(mp4_path))[0]  # 예: "고속철도_220916_영암1"
        # prefix = f"{mp4_stem}_frame_{frame_index:06d}"
        mp4_stem = cfg.source_stem or os.path.splitext(os.path.basename(mp4_path))[0]
        prefix = f"{mp4_stem}_frame_{frame_index:06d}"

        img_name = f"{prefix}.jpg"
        # img_path = os.path.join(frames_dir, img_name)
        # cv2.imwrite(img_path, annotated)
        # 원본 저장 (bbox 그리기 전)
        origin_path = os.path.join(origin_dir, img_name)
        cv2.imwrite(origin_path, frame)  # frame이 원본 프레임(보통 BGR)

        # bbox 그린 결과 저장
        detect_path = os.path.join(detect_dir, img_name)
        cv2.imwrite(detect_path, annotated)

        # JSON 생성
        dets: List[Dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            use_idx = det_indices if cfg.keep_classes is not None else range(len(cls))
            for i in use_idx:
                cls_id = int(cls[i])
                raw_label = (names_map.get(cls_id) if names_map else _get_cls_name(r.names, cls_id))
                final_label = _remap_name(raw_label)  # 예: "고속철도_FAST clip_훼손"
                parts = final_label.split("_")
                rail_type = parts[0].strip() if len(parts) >= 1 else ""
                cls_name = parts[1].strip() if len(parts) >= 2 else ""
                detail = "_".join(parts[2:]).strip() if len(parts) >= 3 else ""
                dets.append({
                    "cls_id": cls_id,
                    "rail_type": rail_type,  # 예: "고속철도"
                    "cls_name": cls_name,  # 예: "FAST clip"
                    "detail": detail,  # 예: "훼손" / "너트 풀림"
                    "confidence": float(confs[i]),
                    "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
                })

        js = {
            "source_mp4": os.path.basename(mp4_path),
            "frame_index": int(frame_index),
            "timestamp_ms": float(timestamp_ms),
            "image_file": img_name,
            "detections": dets,
            "is_anomaly": bool(len(dets) > 0) if cfg.keep_classes is not None else bool(has_det),
        }

        json_name = f"{prefix}.json"
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)

        items.append(js)
        saved += 1
        frame_index += 1

    cap.release()

    summary = {
        "mp4": mp4_path,
        "weights": cfg.weights_path,
        "names_overridden_by_yaml": bool(names_map),
        "fps": fps,
        "total_frames": total_frames,
        "stride": cfg.stride,
        "processed_frames": processed,
        "saved_frames": saved,
        "elapsed_sec": round(time.time() - t0, 3),
        "output_dir": os.path.abspath(cfg.out_dir),
        "origin_dir": os.path.abspath(origin_dir),
        "detect_dir": os.path.abspath(detect_dir),
        "json_dir": os.path.abspath(json_dir),
        "items_count": len(items),
    }
    return summary


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent  # infer.py가 있는 폴더 = Yolo-v8n-laf 루트

    # ✅ 기본값은 "상대경로" (루트 기준)
    default_weights = ROOT / "best.pt"
    default_data_yaml = ROOT / "insulator_hs_ns.yaml"
    default_out_dir = ROOT / "outputs_infer"

    # ✅ 환경변수로 덮어쓰기 가능 (서버/AWS에서 경로 바뀌어도 여길 안 고침)
    MP4_PATH = os.getenv("MP4_PATH", "")  # 로컬 테스트용(비워도 됨, FastAPI에선 mp4_path를 함수로 넣을 거라 상관없음)
    WEIGHTS = os.getenv("WEIGHTS_PATH", str(default_weights))
    DATA_YAML = os.getenv("DATA_YAML_PATH", str(default_data_yaml))
    OUT_DIR = os.getenv("OUT_DIR", str(default_out_dir))

    # 옵션도 환경변수로 조절 가능하게(필요 없으면 고정해도 됨)
    CONF = float(os.getenv("CONF", "0.25"))
    IOU = float(os.getenv("IOU", "0.7"))
    STRIDE = int(os.getenv("STRIDE", "5"))
    SAVE_ALL = os.getenv("SAVE_ALL", "0") == "1"

    cfg = InferConfig(
        weights_path=WEIGHTS,
        data_yaml_path=DATA_YAML,   # 클래스명 yaml로 덮어쓰기
        out_dir=OUT_DIR,
        conf=CONF,
        iou=IOU,
        stride=STRIDE,
        save_all=SAVE_ALL,
        keep_classes=None,
    )

    if not MP4_PATH:
        raise RuntimeError(
            "MP4_PATH 환경변수가 비어있습니다. 예: set MP4_PATH=D:\\path\\video.mp4"
        )

    summary = infer_mp4(MP4_PATH, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

