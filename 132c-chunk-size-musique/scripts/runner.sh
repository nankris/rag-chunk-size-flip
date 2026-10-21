#!/bin/bash
set -euo pipefail
cd /content/b10-132c
mkdir -p results figures

echo "=== RESUMING EXPERIMENT 132c ON COLAB L4 ===" >> results/job.log
nvidia-smi >> figures/colab_run.txt
printf 'RUN_RESUME %s\n' "$(date -Is)" >> figures/colab_run.txt

echo "=== 2. Qwen3-4B Generation & Judging (Resuming) ===" >> results/job.log
python -u scripts/generate.py confirm >> results/job.log 2>&1

printf 'RUN_END %s\n' "$(date -Is)" >> figures/colab_run.txt
nvidia-smi >> figures/colab_run.txt
echo "=== EXPERIMENT 132c COMPLETE ===" >> results/job.log
touch results/DONE
