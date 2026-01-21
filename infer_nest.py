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
  _norm_label("niaocao"): "공통_조류둥지_탐지",      # 예시
  _norm_label("suliaodai"): "공통_비닐봉투_탐지",
  _norm_label("piaofuwu"): "공통_부유물_탐지",
  _norm_label("qiqiu"): "공통_풍선_탐지",
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


def _get_korean_font(font_size: int = 24):
    # Windows 기본 한글 폰트 (대부분 존재)
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",    # 맑은 고딕
        r"C:\Windows\Fonts\malgunsl.ttf",  # 맑은 고딕 Semilight
        r"C:\Windows\Fonts\gulim.ttc",     # 굴림
        r"C:\Windows\Fonts\batang.ttc",    # 바탕
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, font_size)
    # 폰트 못 찾으면 기본 폰트(한글은 깨질 수 있음)
    return ImageFont.load_default()


from PIL import Image, ImageDraw, ImageFont

def _get_korean_font(font_size: int = 28):
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunsl.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        r"C:\Windows\Fonts\batang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, font_size)
    return ImageFont.load_default()

def draw_boxes_with_remap(frame, boxes, names_obj):
    """
    - 클래스별 색상 다르게
    - bbox 두께 크게
    - 라벨 폰트 크게 + 배경(반투명 느낌) + padding
    """
    if boxes is None or len(boxes) == 0:
        return frame

    # OpenCV(BGR) -> PIL(RGB)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))  # 투명 오버레이
    draw = ImageDraw.Draw(overlay)

    font = _get_korean_font(font_size=30)   # 글자 크기 키움
    bbox_thick = 4                          # 박스 두께 키움

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    W, H = pil.size
    placed = []  # 이미 배치된 라벨 박스들 (x0,y0,x1,y1)

    for i in range(len(clss)):
        cls_id = int(clss[i])
        raw = _get_cls_name(names_obj, cls_id)
        name = _remap_name(raw)  # 예: 고속철도_FAST clip_훼손
        conf = float(confs[i])

        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        # 클래스별 색
        r, g, b = color_for_cls(cls_id)

        # bbox (RGBA) - 여러번 그려서 두께 효과
        for t in range(bbox_thick):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 255), width=1)

        # 라벨 텍스트 (원하면 아래를 띄어쓰기 형태로 바꿔도 됨)
        text = f"{name} {conf:.2f}"

        # 텍스트 크기 계산
        l, t, rr, bb = draw.textbbox((0, 0), text, font=font)
        tw, th = (rr - l), (bb - t)

        pad_x, pad_y = 10, 6
        tx = x1
        ty = y1 - (th + pad_y * 2) - 2
        if ty < 0:
            ty = y1 + 2

        # ✅ 겹치면 아래로 밀기 (간단하지만 효과 좋음)
        def _overlap(a, b):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

        max_tries = 30
        step = th + pad_y * 2 + 2

        for _ in range(max_tries):
            x0, y0 = tx, ty
            x1b = min(tx + tw + pad_x * 2, W - 1)
            y1b = min(ty + th + pad_y * 2, H - 1)
            cand = (x0, y0, x1b, y1b)

            hit = False
            for p in placed:
                if _overlap(cand, p):
                    hit = True
                    break

            if not hit:
                placed.append(cand)
                break

            # 겹치면 아래로 한 칸 내리기
            ty += step
            if ty >= H - (th + pad_y * 2) - 1:
                # 화면 아래 넘어가면 다시 위쪽으로 시도(박스 위)
                ty = max(0, y1 - (th + pad_y * 2) - 2)
                # 그래도 계속 겹치면 그냥 현재 위치에 둠
                # (max_tries 끝나면 cand는 마지막 계산값)

        # 라벨 배경 박스 (반투명)
        x0, y0, x1b, y1b = placed[-1] if placed else (tx, ty, min(tx + tw + pad_x * 2, W - 1), min(ty + th + pad_y * 2, H - 1))


        # 배경은 클래스색을 어둡게/반투명으로
        draw.rectangle([x0, y0, x1b, y1b], fill=(r, g, b, 110))

        # 글자(흰색) + 외곽선(검정)으로 가독성
        text_x = x0 + pad_x
        text_y = y0 + pad_y
        # stroke_width는 PIL버전에 따라 지원됨. 안되면 아래 4방향 그림자로 대체 가능
        try:
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
        except TypeError:
            # fallback: 간단한 그림자
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                draw.text((text_x+dx, text_y+dy), text, font=font, fill=(0,0,0,255))
            draw.text((text_x, text_y), text, font=font, fill=(255,255,255,255))

    # 오버레이 합성 후 BGR로 반환
    out = Image.alpha_composite(pil, overlay).convert("RGB")
    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out_bgr


# def draw_boxes_with_remap(frame, boxes, names_obj):
#     """
#     OpenCV frame(BGR)에 bbox는 cv2로 그리고,
#     텍스트(한글)는 PIL로 그린 뒤 다시 BGR로 변환해서 반환
#     """
#     img = frame.copy()
#     if boxes is None or len(boxes) == 0:
#         return img
#
#     xyxy = boxes.xyxy.cpu().numpy()
#     confs = boxes.conf.cpu().numpy()
#     clss = boxes.cls.cpu().numpy().astype(int)
#
#     # 1) bbox는 OpenCV로 먼저 그림
#     for i in range(len(clss)):
#         x1, y1, x2, y2 = map(int, xyxy[i].tolist())
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
#
#     # 2) 텍스트는 PIL로 (한글 지원)
#     pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil)
#     font = _get_korean_font(font_size=24)
#
#     for i in range(len(clss)):
#         cls_id = int(clss[i])
#         raw = _get_cls_name(names_obj, cls_id)
#         name = _remap_name(raw)  # 예: "고속철도_FAST clip_훼손"
#         conf = float(confs[i])
#
#         x1, y1, x2, y2 = map(int, xyxy[i].tolist())
#         text = f"{name} {conf:.2f}"
#
#         # 텍스트 배경 박스(가독성)
#         tx, ty = x1, max(0, y1 - 28)
#         # 정확한 bbox 계산 (left, top, right, bottom)
#         l, t, r, b = draw.textbbox((0, 0), text, font=font)
#         tw, th = (r - l), (b - t)
#
#         pad_x, pad_y = 4, 3
#         x0, y0 = tx, ty
#         x1, y1 = tx + tw + pad_x * 2, ty + th + pad_y * 2
#
#         # 배경(검정) + 글자(노랑)
#         draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
#         draw.text((tx + pad_x, ty + pad_y), text, font=font, fill=(255, 255, 0))
#
#     out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
#     return out

# def draw_boxes_with_remap(frame, boxes, names_obj):
#     img = frame.copy()
#     if boxes is None or len(boxes) == 0:
#         return img
#
#     xyxy = boxes.xyxy.cpu().numpy()
#     confs = boxes.conf.cpu().numpy()
#     clss = boxes.cls.cpu().numpy().astype(int)
#
#     for i in range(len(clss)):
#         cls_id = int(clss[i])
#         raw = _get_cls_name(names_obj, cls_id)
#         name = _remap_name(raw)
#
#         x1, y1, x2, y2 = map(int, xyxy[i].tolist())
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
#         cv2.putText(
#             img, f"{name} {float(confs[i]):.2f}",
#             (x1, max(0, y1 - 6)),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
#         )
#     return img


def infer_mp4(mp4_path: str, cfg: InferConfig) -> Dict[str, Any]:
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"mp4 not found: {mp4_path}")
    if not os.path.exists(cfg.weights_path):
        raise FileNotFoundError(f"weights not found: {cfg.weights_path}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    frames_dir = os.path.join(cfg.out_dir, "frames")
    json_dir = os.path.join(cfg.out_dir, "json")
    os.makedirs(frames_dir, exist_ok=True)
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
        # annotated = draw_boxes_with_remap(frame, r.boxes, r.names)
        # annotated JPG 저장: keep_classes가 있으면 해당 박스만 그리기
        boxes_to_draw = r.boxes
        if cfg.keep_classes is not None and boxes_to_draw is not None and len(boxes_to_draw) > 0:
            # det_indices는 이미 위에서 keep_classes로 필터링된 index 목록
            boxes_to_draw = boxes_to_draw[det_indices]  # Ultralytics Boxes는 인덱싱 지원

        annotated = draw_boxes_with_remap(frame, boxes_to_draw, r.names)
        # mp4_stem = os.path.splitext(os.path.basename(mp4_path))[0]  # 예: "고속철도_220916_영암1"
        # prefix = f"{mp4_stem}_frame_{frame_index:06d}"
        mp4_stem = cfg.source_stem or os.path.splitext(os.path.basename(mp4_path))[0]
        prefix = f"{mp4_stem}_frame_{frame_index:06d}"

        img_name = f"{prefix}.jpg"
        img_path = os.path.join(frames_dir, img_name)
        cv2.imwrite(img_path, annotated)

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
        "frames_dir": os.path.abspath(frames_dir),
        "json_dir": os.path.abspath(json_dir),
        "items_count": len(items),
    }
    return summary


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent  # infer.py가 있는 폴더 = Yolo-v8n-laf 루트

    # ✅ 기본값은 "상대경로" (루트 기준)
    default_weights = ROOT / "best.pt"
    default_data_yaml = ROOT / "dataset.yaml"
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

