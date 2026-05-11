FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.huggingface \
    FF_SAMPLE_DIR=/app/sample

# System deps for Pillow / OpenCV-style image work
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libwebp7 libtiff6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source + samples (models pulled from separate HF model repo
# at build time so the Space repo stays under the 1 GB LFS quota).
COPY server /app/server
COPY web    /app/web
COPY sample /app/sample

# Pull model checkpoints from companion HF model repo. Override at build time
# with `--build-arg MODELS_REPO=<user>/<repo>` if you fork the demo.
ARG MODELS_REPO=turbosun/fabricflow-models
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${MODELS_REPO}', repo_type='model', \
                      local_dir='/app/runs', \
                      allow_patterns=['*.pth'])"

# HF Spaces convention: app must listen on $PORT (default 7860)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
