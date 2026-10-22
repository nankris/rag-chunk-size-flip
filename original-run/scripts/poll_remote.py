from pathlib import Path
import subprocess,os,time,shutil,tempfile
p=Path('/content/rag-hotpotqa')
print('UTC',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
if (p/'job.pid').exists():
 pid=int((p/'job.pid').read_text());print(subprocess.getoutput(f'ps -p {pid} -o pid,etime,args'))
for name in ['run.log'] + ([] if (p/'run.log').exists() and 'STAGE_START' in (p/'run.log').read_text(errors='replace') else ['results/install.log']):
 f=p/name
 if f.exists():print(name,'\n','\n'.join(f.read_text(errors='replace').splitlines()[-12:]))
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'))
print('FILES',[(f.name,f.stat().st_size) for f in (p/'results').glob('*.jsonl')])
if (p/'results').exists():
 with tempfile.TemporaryDirectory(prefix='rag-hotpotqa-snapshot-') as snap:
  for dirname in ['results','figures']:
   shutil.copytree(p/dirname,Path(snap)/dirname,ignore=shutil.ignore_patterns('*.npy','*.part*'))
  shutil.copyfile(p/'run.log',Path(snap)/'run.log')
  subprocess.run(['tar','-czf','/content/rag-hotpotqa-checkpoint.tar.gz','-C',snap,'results','figures','run.log'],check=True)
print('DONE_EXISTS',(p/'results/DONE').exists())
