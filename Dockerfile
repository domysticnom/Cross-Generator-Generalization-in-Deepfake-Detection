# Reproducible CPU image for the project. On a GPU box, install torch from the
# cu128 index instead (see README.md). Build and run:
#   docker build -t deepfake-xgen .
#   docker run --rm deepfake-xgen        # runs the environment smoke test
FROM python:3.11-slim

# ffmpeg: video decoding. libgl1 + libglib2.0-0: opencv runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# torch first (CPU wheels; requirements.txt omits it by design), then the rest.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default: environment smoke test (torch, ffmpeg, core imports).
CMD ["python", "check_env.py"]
