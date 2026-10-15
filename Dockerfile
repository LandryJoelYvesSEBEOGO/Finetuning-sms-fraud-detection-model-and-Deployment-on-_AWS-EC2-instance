# syntax=docker/dockerfile:1

# --- Build stage: install dependencies into an isolated virtualenv ---------
FROM python:3.12-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
# CPU-only torch first, so requirements.txt does not pull the CUDA build.
RUN pip install torch==2.5.1 --index-url "$TORCH_INDEX_URL" \
 && pip install -r requirements.txt

# --- Runtime stage -----------------------------------------------------------
FROM python:3.12-slim

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN useradd --create-home --user-group appuser
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY sms_fraud/ ./sms_fraud/

USER appuser
RUN mkdir -p "$HF_HOME"
EXPOSE 8080

# Uvicorn only starts listening once the model is loaded (lifespan), which
# includes downloading it on first start: hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=4)"]

CMD ["uvicorn", "sms_fraud.main:app", "--host", "0.0.0.0", "--port", "8080"]
