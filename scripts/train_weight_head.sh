#!/usr/bin/env bash
set -euo pipefail
python -m weight_head.train --config "${1:-configs/weight_head.yaml}"
