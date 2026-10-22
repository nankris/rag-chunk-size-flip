#!/bin/bash
set -euo pipefail
SESSION="rag-musique"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Colab Usage ==="
colab usage || true

echo "=== Remote Log Tail ==="
colab exec -s "$SESSION" << 'PYEOF'
import os
log_file = '/content/rag-musique/results/job.log'
if os.path.exists(log_file):
    lines = open(log_file).read().splitlines()
    for l in lines[-20:]:
        print(l)
else:
    print('No log file yet.')
PYEOF

echo "=== Syncing Results Back ==="
colab exec -s "$SESSION" << 'PYEOF'
import os, subprocess
p = '/content/rag-musique/results'
if os.path.exists(p):
    files = [f for f in os.listdir(p) if f.endswith('.jsonl') or f.endswith('.json') or f == 'DONE' or f == 'job.log']
    print('Remote result files:', len(files), 'files found')
    cmd = "tar --warning=no-file-changed --exclude='*.npy' -czf /content/results_sync.tar.gz -C /content/rag-musique results figures || true"
    subprocess.call(cmd, shell=True)
PYEOF

rm -f /tmp/results_sync.tar.gz
colab download -s "$SESSION" /content/results_sync.tar.gz /tmp/results_sync.tar.gz
if [ -f /tmp/results_sync.tar.gz ]; then
    tar xzf /tmp/results_sync.tar.gz -C "$ROOT"
    echo "Synced results and figures to local directory."
fi
