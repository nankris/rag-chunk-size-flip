import subprocess,os
from pathlib import Path
p=Path('/content/rag-hotpotqa');p.mkdir(exist_ok=True)
if (p/'job.pid').exists():
    pid=(p/'job.pid').read_text().strip()
    cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and b'scripts/run.sh' in cmd.read_bytes():
        print('JOB_ALREADY_RUNNING',pid);raise SystemExit(0)
subprocess.run(['tar','-xzf','/content/rag-hotpotqa-scripts.tar.gz','-C',str(p)],check=True)
log=(p/'run.log').open('a')
proc=subprocess.Popen(['nohup','bash','scripts/run.sh'],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(p/'job.pid').write_text(str(proc.pid))
print('DETACHED_PID',proc.pid)
