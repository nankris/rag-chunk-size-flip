"""Retry only our named session at five-minute intervals; log every attempt."""
import subprocess,time,datetime,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'results/allocation_attempts.jsonl'
for attempt in range(25):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    r=subprocess.run(['colab','new','--gpu','L4','-s','b8-132'],capture_output=True,text=True)
    msg=r.stdout+r.stderr
    with p.open('a') as f:f.write(json.dumps({'utc':started,'attempt':attempt+1,'returncode':r.returncode,'output':msg})+'\n')
    print(started,msg,flush=True)
    # CLI may return zero for a refused allocation. Verify our named session.
    status=subprocess.run(['colab','status','-s','b8-132'],capture_output=True,text=True)
    if 'not found' not in (status.stdout+status.stderr).lower() and status.returncode==0:
        print('SESSION_READY',flush=True);break
    if attempt==24:raise SystemExit('Allocation unavailable after two hours')
    time.sleep(300)
