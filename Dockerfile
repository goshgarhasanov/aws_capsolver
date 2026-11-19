# aws_capsolver - CPU-only image.
#
#   docker build -t aws-capsolver .
#   docker run -d -p 8790:8790 -e AWSCS_KEYS=MY-KEY -v awscs-models:/models aws-capsolver
#
# Backbone weights (open_clip / faster-whisper, from the Hugging Face hub) are not baked in:
# they are downloaded into /models on first use, so mount a volume there to keep them.
# The trained head ships in /app/weights; mount a directory over it to use a newer one.
FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG APP_UID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache \
    TORCH_HOME=/models/torch \
    AWSCS_KEYS_FILE=/config/keys.json \
    AWSCS_USAGE_DB=/state/usage.db

# ffmpeg: audio CAPTCHA decoding tools; tini: proper signal handling as PID 1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first so later installs do not pull the CUDA wheels
RUN pip install --index-url "${TORCH_INDEX_URL}" torch torchvision

COPY requirements.txt ./
RUN grep -vE '^\s*(#|$)|playwright|pytest|httpx' requirements.txt > /tmp/req-runtime.txt \
 && pip install -r /tmp/req-runtime.txt \
 && pip install numpy open_clip_torch faster-whisper av scipy \
 && rm /tmp/req-runtime.txt

COPY aws_capsolver ./aws_capsolver
COPY weights ./weights
COPY pyproject.toml README.md ./

RUN useradd --system --uid "${APP_UID}" --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /models /config /state \
 && chown -R app:app /models /state /app/weights

USER app
VOLUME ["/models", "/state"]
EXPOSE 8790

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import json,sys,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8790/health', timeout=4); sys.exit(0 if json.load(r).get('ok') else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
# exactly one process: the task registry is in memory; scale with AWSCS_WORKERS, never --workers
CMD ["python", "-m", "aws_capsolver", "serve", "--host", "0.0.0.0", "--port", "8790"]
