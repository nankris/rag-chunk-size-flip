#!/bin/bash
set -euo pipefail
cd /content/132b
mkdir -p results
nvidia-smi
date -Is
pip freeze > results/packages.txt
for stage in prepare retrieve generate; do
  echo "STAGE_START $stage $(date -Is)"
  python -u scripts/$stage.py
  echo "STAGE_END $stage $(date -Is)"
done
