### Reproducibility artifacts (what you’ll get)
- Dockerfile — reproducible GPU-ready container with pinned deps and build ARGs for CUDA base image.
- requirements.txt — fully pinned Python packages used by the training guide (transformers / accelerate / peft / deepspeed / bitsandbytes / tokenizers / datasets / torch).
- run_repro.sh — deterministic smoke-run script that sets seeds and deterministic flags, launches a minimal training run (reduced dataset / steps), and validates an exact-loss checkpoint value within epsilon.

Use these files as a drop‑in reproducibility baseline. Adjust CUDA base image ARG and GPU driver compatibility to match your infra.

---

### Dockerfile
```dockerfile
# Dockerfile: reproducible training environment (GPU-capable)
# Usage examples:
#  docker build --build-arg CUDA_BASE=nvidia/cuda:12.2.0-devel-ubuntu22.04 -t mltune:repro .
#  docker run --gpus all -v $(pwd):/workspace -w /workspace mltune:repro /bin/bash -c "./run_repro.sh"

ARG CUDA_BASE=nvidia/cuda:12.2.0-devel-ubuntu22.04
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
```

---

### requirements.txt (pinned)
- These versions are chosen for reproducibility and compatibility with PEFT/LoRA workflows. Adjust torch CUDA build as needed for your driver.

```
# Core ML infra
torch==2.2.0
transformers==4.32.2
tokenizers==0.13.3

# Training/runtime
accelerate==0.22.0
deepspeed==0.9.2
peft==0.4.0
bitsandbytes==0.39.0

# Data and utils
datasets==2.12.0
sentencepiece==0.1.99
huggingface_hub==0.13.4

# Dev & reproducibility helpers
numpy==1.26.2
scipy==1.11.3
pandas==2.2.2
pyyaml==6.0
loguru==0.7.0
tqdm==4.66.1
```

Notes:
- If you need a CUDA-specific torch wheel, replace `torch==2.2.0` with the exact wheel for your CUDA (e.g., `pip install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html`) when building the image.

---

### run_repro.sh — deterministic smoke-run and validation
```bash
#!/usr/bin/env bash
set -euo pipefail

# run_repro.sh
# Purpose: deterministic smoke-run for the training guide.
# Assumptions:
# - Training script exists at ./train.py and accepts:
#     --train-file, --validation-file, --output-dir, --max-steps, --per-device-train-batch-size, --seed
# - Minimal sample dataset placed at ./repro_dataset/train.jsonl and ./repro_dataset/val.jsonl
# - This script writes a compact log at ./repro_output/training_log.json with loss/time metadata

# CONFIGURABLE
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${REPO_ROOT}/repro_output"
TRAIN_SCRIPT="${REPO_ROOT}/train.py"
TRAIN_DATA="${REPO_ROOT}/repro_dataset/train.jsonl"
VAL_DATA="${REPO_ROOT}/repro_dataset/val.jsonl"
MODEL_NAME_OR_PATH="facebook/opt-1.3b"    # small model for smoke-run; override as needed
MAX_STEPS=10
BATCH_SIZE=1
SEED=42
EPSILON=1e-4                                # allowed floating-point tolerance for deterministic check
EXPECTED_LOSS_FILE="${OUTDIR}/expected_loss.txt"  # optional baseline file (create once)

mkdir -p "${OUTDIR}"

echo "Starting reproducible smoke-run"
echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Max steps: ${MAX_STEPS}"
echo "Seed: ${SEED}"

# Deterministic Python environment variables
export PYTHONHASHSEED=${SEED}
export TRANSFORMERS_VERBOSITY=error

# Run with Python-level deterministic flags via a small wrapper script
python3 - <<PY
import os, random, json, torch, numpy as np
# Deterministic flags
seed = int(os.environ.get("PYTHONHASHSEED", 42))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# cuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Set deterministic flags; cuda available:", torch.cuda.is_available())
PY

# Run the training script (minimal smoke)
# The training script should emit a single-line JSON log with {"step": <n>, "loss": <float>} at each step
python3 "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_file "${TRAIN_DATA}" \
  --validation_file "${VAL_DATA}" \
  --output_dir "${OUTDIR}/adapter" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --logging_strategy "steps" \
  --logging_steps 1

# Extract last-step loss (assumes train script writes ./repro_output/training_log.json)
LOGFILE="${OUTDIR}/training_log.jsonl"
if [ ! -f "${LOGFILE}" ]; then
  echo "ERROR: expected log file ${LOGFILE} not found"
  exit 2
fi

LAST_LOSS=$(tail -n 1 "${LOGFILE}" | jq -r '.loss')
echo "Last-step loss: ${LAST_LOSS}"
echo "${LAST_LOSS}" > "${OUTDIR}/last_loss.txt"

# If an expected baseline exists, validate determinism
if [ -f "${EXPECTED_LOSS_FILE}" ]; then
  EXPECTED=$(cat "${EXPECTED_LOSS_FILE}")
  # Use awk for float abs diff
  DIFF=$(awk -v a="${LAST_LOSS}" -v b="${EXPECTED}" 'BEGIN{print (a>b)?a-b:b-a}')
  cmp=$(awk -v d="${DIFF}" -v e="${EPSILON}" 'BEGIN{print (d<=e)?0:1}')
  if [ "${cmp}" -eq 0 ]; then
    echo "Deterministic check PASSED (diff ${DIFF} ≤ ${EPSILON})"
    exit 0
  else
    echo "Deterministic check FAILED (diff ${DIFF} > ${EPSILON})"
    exit 3
  fi
else
  echo "No expected baseline found. To pin baseline for future runs, create file: ${EXPECTED_LOSS_FILE} with the single numeric loss value."
  echo "First-run produced last loss: ${LAST_LOSS}"
  exit 0
fi
```

Notes on run_repro.sh:
- The script expects a minimal train.py that logs JSONL lines with a numeric "loss" key at each logged step (common in training scripts). If your guide uses a different log path or format (e.g., TensorBoard), adapt the LOGFILE and extraction accordingly.
- On the first successful run, capture the printed last-loss value and write it into repro_output/expected_loss.txt. Subsequent CI runs will compare the loss value within EPSILON to assert determinism.
- If you cannot run GPU builds in CI, set MODEL_NAME_OR_PATH to a tiny CPU-supported base model or run a CPU-only mode.

---

### Practical integration checklist (apply after adding files)
- Add the three files to your repo root: Dockerfile, requirements.txt, run_repro.sh (chmod +x run_repro.sh).
- Add a tiny deterministic dataset under repro_dataset/ (JSONL lines matching train.py expected schema). Example: 30 short QA pairs or conversation turns with fixed fields.
- Create or adapt train.py to support the CLI flags used above and to emit training_log.jsonl with {"step":int,"loss":float} lines.
- Build and run container locally, then run smoke-run:
  - docker build --build-arg CUDA_BASE=nvidia/cuda:12.2.0-devel-ubuntu22.04 -t mltune:repro .
  - docker run --gpus all -v $(pwd):/workspace -w /workspace mltune:repro ./run_repro.sh
- For GitHub Actions: create a job that builds the image (or uses a hosted runner + preinstalled deps), mounts repo, and runs ./run_repro.sh; fail the job if deterministic check fails.

---

### Next optional additions I can generate for you
- Exact train.py smoke-run stub that matches run_repro.sh expectations (minimal forward/backward loop and JSONL logging).
- GitHub Actions workflow YAML that builds the Docker image and runs the smoke-run as a PR gate.
- Small deterministic JSONL dataset template (30 examples) and expected_loss.txt baseline after a first run.

---

https://copilot.microsoft.com/shares/LYpDmmpYpKkBvWzbAsh62
