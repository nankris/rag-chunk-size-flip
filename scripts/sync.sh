#!/bin/bash
set -euo pipefail
cd /workspace/work/tig/132-rag-chunk-size-flip
source /workspace/tools/setup.sh
colab usage | tee -a results/usage.txt
colab exec -s b8-132 --timeout 25 -f scripts/poll_remote.py
colab download -s b8-132 /content/b8-132-checkpoint.tar.gz checkpoint.tar.gz
tar -xzf checkpoint.tar.gz
