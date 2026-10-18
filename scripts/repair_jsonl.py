"""Recover complete rows if an interrupted append left an incomplete final line."""
import json
from pathlib import Path
def repair(root):
 for p in Path(root).glob('*.jsonl'):
  raw=p.read_bytes();lines=raw.splitlines(keepends=True);good=[]
  for i,line in enumerate(lines):
   try:json.loads(line)
   except (ValueError,UnicodeError):
    if i!=len(lines)-1:raise RuntimeError(f'Non-tail corruption: {p}:{i+1}')
    print('Truncating incomplete tail',p,flush=True);break
   good.append(line if line.endswith(b'\n') else line+b'\n')
  fixed=b''.join(good)
  if fixed!=raw:p.write_bytes(fixed)
if __name__=='__main__':repair(Path(__file__).resolve().parents[1]/'results')
