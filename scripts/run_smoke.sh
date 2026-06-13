#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/kxh/miniconda3/envs/trl/bin/python}"
ANNOTATIONS_JSONL="${ANNOTATIONS_JSONL:-/mnt/kxh/smx/homework/h3/datasets/flickr30k/annotations.jsonl}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
"${PYTHON_BIN}" scripts/evaluate_flickr30k.py \
  --annotations-jsonl "${ANNOTATIONS_JSONL}" \
  --clip-model openai/clip-vit-base-patch32 \
  --longclip-checkpoint checkpoints/longclip-B.pt \
  --max-images 32 \
  --output-dir results/smoke
