# Dockerfile: reproducible training environment (GPU-capable)
# Usage examples:
#  docker build --build-arg CUDA_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04 -t mltune:repro .
#  docker run --gpus all -v $(pwd):/workspace -w /workspace mltune:repro /bin/bash -c "./run_repro.sh"

ARG CUDA_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Basic deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates python3 python3-dev python3-pip python3-venv \
    libsndfile1 ffmpeg locales tzdata && \
    rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy project skeleton (expect to run build context with repo root)
WORKDIR /workspace
COPY . /workspace

# Install pinned Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Create a reproducible non-root user (optional)
RUN useradd -ms /bin/bash runner && chown -R runner:runner /workspace
USER runner

# Entrypoint left minimal; use run_repro.sh directly
ENV PATH="/home/runner/.local/bin:${PATH}"
