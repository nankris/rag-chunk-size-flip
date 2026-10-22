#!/bin/bash
set -euo pipefail
cd /content/rag-musique
mkdir -p results figures

echo "=== RESUMING MUSIQUE CHUNK-SIZE STUDY ON COLAB L4 ===" >> results/job.log
nvidia-smi >> figures/colab_run.txt
printf 'RUN_RESUME %s\n' "$(date -Is)" >> figures/colab_run.txt

echo "=== 2. Qwen3-4B Generation & Judging (Resuming) ===" >> results/job.log
python -u scripts/generate.py confirm >> results/job.log 2>&1

printf 'RUN_END %s\n' "$(date -Is)" >> figures/colab_run.txt
nvidia-smi >> figures/colab_run.txt
echo "=== MUSIQUE CHUNK-SIZE STUDY COMPLETE ===" >> results/job.log
touch results/DONE
