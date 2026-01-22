# PikNLP-inference 참고: pytorch runtime + uvicorn 구동
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 시스템 패키지 (OpenCV/영상 처리에 자주 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-nanum \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workdir

# 1) 서버 의존성 먼저 복사/설치 (캐시 최적화)
COPY requirements.api.txt /workdir/requirements.api.txt
RUN pip install --no-cache-dir -r /workdir/requirements.api.txt

# 2) 프로젝트 전체 복사
COPY . /workdir

# 3) "커스텀 ultralytics"를 editable로 강제 설치 (중요!)
#    -> 절대 원본 ultralytics 쓰지 않게 보장
RUN pip uninstall -y ultralytics || true \
 && pip install --no-cache-dir -e /workdir/ultralytics_custom/ultralytics-8.4.6

# 포트
EXPOSE 8000

# 실행
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]