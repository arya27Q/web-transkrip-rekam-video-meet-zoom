FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOXSCRIBE_DATA_DIR=/app/data \
    HF_HOME=/app/hf_cache

WORKDIR /app

# Install dependency inti dengan wheel biner (tanpa memerlukan FFmpeg eksternal)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    pydantic \
    python-multipart \
    youtube-transcript-api \
    ctranslate2 \
    huggingface-hub \
    tokenizers \
    numpy \
    tqdm && \
    pip install --no-cache-dir --no-deps faster-whisper

COPY main.py database.py audio_service.py exporters.py index.html ./

RUN mkdir -p /app/data/audio /app/hf_cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
