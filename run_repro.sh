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
