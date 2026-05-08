# GPU image for RunPod / any NVIDIA host (~6GB+ VRAM for typical Hindi Whisper + fp16).
# Pick a PyTorch tag that matches your host CUDA; see https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app.py clinic_workflow.py textbook_rag.py ./
COPY data ./data

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    ASR_MODEL_DIR=/workspace/models/whisper-hindi2hinglish-prime \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

EXPOSE 8080

# First model load can exceed 30s; adjust start-period if cold-start is slower.
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health >/dev/null || exit 1

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 120"]
