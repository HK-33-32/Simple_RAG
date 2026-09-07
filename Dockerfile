FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 15 \
        torch==2.14.0 --index-url ${TORCH_INDEX_URL}
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

COPY app.py rag.py rag_ext.py ./
COPY static ./static

RUN mkdir -p /app/data/docs /app/data/uploads /app/vector_db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
