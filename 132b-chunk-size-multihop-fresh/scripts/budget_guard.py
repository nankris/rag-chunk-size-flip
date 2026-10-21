"""Conservative local cap guard; only this experiment's session may be stopped."""
import time,subprocess,json,re,datetime,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
(P/'results/guard.pid').write_text(str(os.getpid()))
while not (P/'results/SESSION_STOPPED').exists():
 try:
  r=subprocess.run(['colab','usage'],capture_output=True,text=True,timeout=45,check=True);m=re.search(r'Current balance:\s*([0-9.]+)',r.stdout)
  if m:
   balance=float(m.group(1));used=127.05-balance
   with (P/'results/guard_usage.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'balance':balance,'account_delta':used})+'\n')
   if used>=5.5:
    (P/'results/CAP_STOP').write_text(f'Shared-account decrease {used:.2f} CU, conservative cutoff 5.5 CU')
    try:subprocess.run(['/workspace/tools/mlenv/bin/python',str(P/'scripts/sync.py')],timeout=100)
    finally:subprocess.run(['colab','stop','-s','b9-132b'],timeout=60)
    (P/'results/SESSION_STOPPED').write_text('budget guard');break
 except Exception as e:print(type(e).__name__,str(e),flush=True)
 time.sleep(60)
