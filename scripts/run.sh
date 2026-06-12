#!/bin/bash
# Unified entry point for train/eval.
#
# Quick start — edit the User Config below, then: bash run.sh
# Or pass args: bash run.sh <config> [mode] [KEY=VAL ...]
#   config: matches configs/<name>.yaml
#   mode:   train (default) or eval
#   KEY=VAL: override any value (e.g. GPU_IDS=0,1)

# ======================== User Config ========================
CONFIG=my_experiment    # experiment name (matches configs/<name>.yaml)
MODE=train              # train or eval
GPU_IDS=0               # e.g. 0,1 for multi-GPU
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -euo pipefail

# CLI args override User Config
if [ $# -gt 0 ]; then
    CONFIG="$1"
    shift
    if [ $# -gt 0 ] && [[ ! "$1" == *=* ]]; then
        MODE="$1"
        shift
    fi
fi

# Load YAML config
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/${CONFIG}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config not found: ${CONFIG_FILE}" >&2
    echo "Available:" && ls "${PROJECT_ROOT}/configs/" | sed 's/\.yaml$//' >&2
    exit 1
fi

source "${PROJECT_ROOT}/dancher_tools/scripts/_parse_yaml.sh"
_parse_yaml "$CONFIG_FILE"

# ── Map YAML keys → shell variables ──────────────────────────
# YAML keys become lowercase shell vars after _parse_yaml.
# Example YAML:
#   model: my_model       →  MODEL="${model}"
#   epochs: 500           →  EPOCHS="${epochs:-500}"
#   batch_size: 4         →  BATCH_SIZE="${batch_size:-4}"
#
# Uncomment and customize:
MODEL="${model:?model not set in ${CONFIG}.yaml}"
# EPOCHS="${epochs:-500}"
# BATCH_SIZE="${batch_size:-4}"
# LR="${lr:-1e-3}"

source "${SCRIPT_DIR}/_common.sh"
