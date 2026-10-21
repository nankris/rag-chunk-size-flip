import pathlib,subprocess,tarfile,os,time
p=pathlib.Path('/content/132b');p.mkdir(exist_ok=True)
with tarfile.open('/content/132b_code.tar.gz') as t:t.extractall(p)
if (p/'results/DONE').exists():
 print('ALREADY_COMPLETE');raise SystemExit(0)
# A rerun never starts a duplicate live job.
pidpath=p/'job.pid';live=False
if pidpath.exists():
 try:
  pid=int(pidpath.read_text());os.kill(pid,0)
  stat=pathlib.Path(f'/proc/{pid}/stat').read_text().split()
  live=stat[2]!='Z'
 except ProcessLookupError:pass
if not live:
 f=open(p/'run.log','a');proc=subprocess.Popen(['bash',str(p/'scripts/run.sh')],stdout=f,stderr=f,start_new_session=True,cwd=p);pidpath.write_text(str(proc.pid));print('LAUNCHED',proc.pid)
else:print('ALREADY_RUNNING',pidpath.read_text())
print(subprocess.getoutput('tail -12 /content/132b/install.log'))
