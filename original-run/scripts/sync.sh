#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
colab --help
python scripts/usage_check.py | tee -a results/usage.txt
timeout 55s colab exec -s rag-hotpotqa --timeout 25 -f scripts/poll_remote.py
timeout 55s colab download -s rag-hotpotqa /content/rag-hotpotqa-checkpoint.tar.gz checkpoint.tar.gz
tar -xzf checkpoint.tar.gz
