"""Retain original terminal output and recovery/accounting evidence."""
from pathlib import Path
P=Path(__file__).resolve().parents[1]
parts=[]
for name in ['run.log','gpu_snapshots.txt','recoveries.jsonl','sync_errors.jsonl','usage_final.json','stop_session.txt']:
    f=P/'results'/name
    if f.exists():parts.extend(['\nFILE: results/'+name+'\n',f.read_text()])
(P/'figures/colab_run.txt').write_text(''.join(parts))
print('WROTE figures/colab_run.txt')
