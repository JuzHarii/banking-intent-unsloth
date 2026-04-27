#!/bin/bash
# ── Train the banking intent classifier ──────────────────────
# Usage: bash train.sh [path/to/train.yaml]

set -e

CONFIG=${1:-configs/train.yaml}

echo "=========================================="
echo " Banking Intent Classifier — Training"
echo " Config: $CONFIG"
echo "=========================================="

python scripts/train.py --config "$CONFIG"

echo ""
echo "✅ Training complete."
echo "   Upload the 'model_trained/' folder to Hugging Face,"
echo "   then update hf_repo in configs/inference.yaml."