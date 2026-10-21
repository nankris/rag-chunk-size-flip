"""Short remote snapshot call; safe to rerun after sandbox reboot."""
import subprocess,json,datetime,tarfile
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def call(args,**kw):
 def decoded(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
 for attempt in range(2):
  try:return subprocess.run(['colab',*args],text=True,capture_output=True,check=True,timeout=35 if args[0]=='exec' else 90,**kw)
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
   out=decoded(getattr(e,'stdout',''));err=decoded(getattr(e,'stderr',''))
   if args[0]=='exec' and 'SNAPSHOT_READY_132b' in out:
    return subprocess.CompletedProcess(args,0,out,err)
   if args[0]!='exec' or attempt or not (isinstance(e,subprocess.TimeoutExpired) or 'Connection was lost' in out+err):raise
   with (P/'results/sync_errors.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':type(e).__name__,'action':'restart named notebook kernel, leaving detached GPU process intact'})+'\n')
   subprocess.run(['colab','restart-kernel','-s','b9-132b'],check=True,capture_output=True,text=True,timeout=45)
u=call(['usage']).stdout
with (P/'results/usage_history.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'usage':u})+'\n')
print(u)
code=r"""
import subprocess,pathlib,tarfile
p=pathlib.Path('/content/132b')
print(subprocess.getoutput('tail -8 /content/132b/run.log'))
gpu=subprocess.getoutput('nvidia-smi')
print('GPU snapshot: '+next((x.strip() for x in gpu.splitlines() if 'NVIDIA L4' in x),'saved'))
with (p/'results/gpu_snapshots.txt').open('a') as f:f.write(subprocess.getoutput('date -Is')+'\n'+gpu+'\n')
with tarfile.open('/content/132b_snapshot.tar.gz','w:gz',compresslevel=1) as t:
 for f in (p/'results').glob('*'):
  if f.suffix in ['.json','.jsonl','.txt','.log'] or f.name=='DONE':t.add(f,arcname='results/'+f.name)
 for name in ['run.log','install.log','job.pid']:
  f=p/name
  if f.exists():t.add(f,arcname='results/'+name)
print('SNAPSHOT_READY_132b',flush=True)
"""
r=call(['exec','-s','b9-132b','--timeout','60'],input=code);print(r.stdout)
call(['download','-s','b9-132b','/content/132b_snapshot.tar.gz',str(P/'results/snapshot.tar.gz')])
with tarfile.open(P/'results/snapshot.tar.gz') as t:t.extractall(P,filter='data')
print('SNAPSHOT_SAVED')
