#!/usr/bin/env bash
set -euo pipefail
python -m grpo.train --config "${1:-configs/grpo_gpt.yaml}"
