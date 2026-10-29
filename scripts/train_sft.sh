#!/usr/bin/env bash
set -euo pipefail
python -m sft.train_schema_sft --config "${1:-configs/sft.yaml}"
