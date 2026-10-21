#!/bin/bash
set -euo pipefail
cd /content/b8-132
mkdir -p results figures
nvidia-smi >> figures/colab_run.txt
printf 'RUN_START %s\n' "$(date -Is)" >> figures/colab_run.txt
python -u -m pip install 'vllm==0.23.0' 'transformers==5.5.4' faiss-cpu > results/install.log 2>&1
python scripts/repair_jsonl.py
python -u scripts/prepare.py
for stage in pilot confirm; do
  printf 'STAGE_START %s %s\n' "$stage" "$(date -Is)" | tee -a figures/colab_run.txt
  python -u scripts/retrieve.py "$stage"
  python -u scripts/generate.py "$stage"
  printf 'STAGE_END %s %s\n' "$stage" "$(date -Is)" | tee -a figures/colab_run.txt
done
pip freeze > results/packages.txt
printf 'RUN_END %s\n' "$(date -Is)" | tee -a figures/colab_run.txt
touch results/DONE
