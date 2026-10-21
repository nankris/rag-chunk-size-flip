"""Local checkpoint copier; GPU inference remains detached on Colab."""
import subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[1]
while not (P/'results/SESSION_STOPPED').exists():
    r=subprocess.run(['/workspace/tools/mlenv/bin/python',str(P/'scripts/sync.py')],cwd=P)
    if (P/'results/DONE').exists():
        print('COMPLETE_SNAPSHOT_READY',flush=True)
        break
    if r.returncode: print('SYNC_FAILED_WILL_RETRY',r.returncode,flush=True)
    time.sleep(120)
