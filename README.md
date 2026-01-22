# RailDock Vision AI – Inference Baseline

철도 시설물 영상(MP4)을 입력으로 받아  
애자, 전차선, 클램프, 프로텍터, 행거 등의 **이상 객체를 자동 탐지**하는  
Vision AI 추론 베이스라인 프로젝트입니다.

YOLOv8 계열을 기반으로 하며,  
논문 *A Lightweight Transmission Line Foreign Object Detection Algorithm Incorporating Adaptive Weight Pooling*  
의 구조를 참고하여 **경량화 + 다중 스케일 탐지 성능**을 강화한 커스텀 모델을 사용합니다.

---

## 1. 프로젝트 개요

### 1.1 목적

- 고속철도 / 일반철도 영상(MP4)을 입력으로 받아 이상 객체 자동 탐지
- 추론 결과를 zip 파일 형태(이미지 + JSON) 형태로 출력

### 1.2 출력 결과

- Bounding Box가 시각화된 이미지 (`.jpg`)
- 프레임 단위 구조화된 JSON 메타데이터
- 클래스, 신뢰도(confidence), 위치 좌표 포함
- 결과는 zip 파일 형태로 반환

---

## 2. 모델 구조 설명

### 2.1 전체 아키텍처 개요

본 모델은 YOLOv8 Backbone을 기반으로 하되,  
철도 설비 환경에 맞게 다음 모듈을 추가·변형하였습니다.

- **FEA (Feature Extraction Attention)**
- **LWM (Lightweight Weight Module)**
- **C2f-SCConv**
- **SPPF (Spatial Pyramid Pooling – Fast)**

다중 스케일 Detect Head  
(80×80 / 40×40 / 20×20)을 사용하여  
소형 객체(애자, 클립, 전차선 부착물 등)에 대한 탐지 성능을 강화했습니다.

### 2.2 네트워크 구조도

![Model Architecture](architecture.png)

### 2.3 핵심 기여점

- Adaptive Weight Pooling 기반 경량화 구조
- Neck 단계에서 다중 해상도 Feature 간 정보 손실 최소화
- 소형 철도 설비 객체 탐지 성능 개선
- 기존 YOLO 계열 대비 연산량 감소 + 정확도 유지

본 프로젝트는 해당 논문의 구조를 참고하여  
YOLOv8 프레임워크 위에 커스텀 모듈 형태로 구현되었습니다.

---

## 3. 디렉토리 구조

```text
baseline/
├── app.py                     # FastAPI 추론 서버 엔트리포인트
├── infer_insulator.py          # 애자(Insulator) 추론 로직
├── infer_nest.py               # 둥지(Nest) 추론 로직
├── infer_rail.py               # 선로/전차선 추론 로직
├── ultralytics_custom/         # 커스텀 YOLOv8 모듈
│   └── ultralytics-8.4.6/
├── insulator/                  # 애자 모델 관련 리소스
├── nest/                       # 둥지 모델 관련 리소스
├── rail/                       # 선로 모델 관련 리소스
├── Dockerfile                  # 추론 서버 Docker 이미지 정의
├── requirements.api.txt        # API / Inference용 Python 패키지
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 4. 사용 방법

본 프로젝트는 다음 두 가지 방식으로 사용할 수 있습니다.

- Docker 기반 FastAPI 추론 서버 실행 (권장)
- 단일 MP4 추론 스크립트 직접 실행

### 4.1 Docker 기반 추론 서버 실행 (권장)
#### Docker 이미지 빌드
```text
docker build -t raildock-vision:latest .
```
#### Docker 컨테이너 실행 (GPU 사용)
```text
docker run --rm --gpus all -p 8000:8000 raildock-vision:latest
```
- FastAPI 서버는 아래 주소에서 실행됩니다.
  - http://localhost:8000

### 4.2 FastAPI 추론 API 사용 예시
#### 엔드포인트
```text
POST /predict_multi3
```
#### 요청 예시 (Windows PowerShell)
```text
curl.exe -X POST "http://127.0.0.1:8000/predict_multi3?stride=5&conf=0.25" `
  -F "rail_mp4=@rail.mp4" `
  -F "insulator_mp4=@insulator.mp4" `
  -F "nest_mp4=@nest.mp4" `
  -o result.zip
```

#### 입력
- MP4 영상 파일 (선로 / 애자 / 둥지)

#### 출력
- Bounding Box가 시각화된 이미지(JPG)
- 프레임 단위 JSON 메타데이터
- 위 결과를 포함한 zip 파일

