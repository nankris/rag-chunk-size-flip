from pathlib import Path
import subprocess,os,time
p=Path('/content/b8-132')
print('UTC',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
if (p/'job.pid').exists():
 pid=int((p/'job.pid').read_text());print(subprocess.getoutput(f'ps -p {pid} -o pid,etime,args'))
for name in ['run.log','results/install.log']:
 f=p/name
 if f.exists():print(name,'\n','\n'.join(f.read_text(errors='replace').splitlines()[-12:]))
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'))
print('FILES',[(f.name,f.stat().st_size) for f in (p/'results').glob('*.jsonl')])
if (p/'results').exists():
 subprocess.run(['tar','-czf','/content/b8-132-checkpoint.tar.gz','--exclude=*.npy','--exclude=*.part*','-C',str(p),'results','figures','run.log'],check=True)
print('DONE_EXISTS',(p/'results/DONE').exists())
