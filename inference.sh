#!/bin/bash
# ── Run inference for the banking intent classifier ──────────
#
# Usage:
#   Single query:   bash inference.sh "I lost my card"
#   Evaluate:       bash inference.sh --evaluate
#   Interactive:    bash inference.sh

set -e

CONFIG=${INFERENCE_CONFIG:-configs/inference.yaml}

if [ "$1" = "--evaluate" ]; then
    echo "Running evaluation on test set..."
    python scripts/inference.py --config "$CONFIG" --evaluate
elif [ -n "$1" ]; then
    python scripts/inference.py --config "$CONFIG" --query "$1"
else
    python scripts/inference.py --config "$CONFIG"
fi