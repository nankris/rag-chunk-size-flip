import subprocess,os
from pathlib import Path
p=Path('/content/b8-132');p.mkdir(exist_ok=True)
subprocess.run(['tar','-xzf','/content/b8-132-scripts.tar.gz','-C',str(p)],check=True)
log=(p/'run.log').open('a')
proc=subprocess.Popen(['bash','scripts/run.sh'],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(p/'job.pid').write_text(str(proc.pid))
print('DETACHED_PID',proc.pid)
