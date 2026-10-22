"""Log shared-account usage and stop only this session before the CU cap."""
import subprocess,json,re,datetime
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'results'
p=subprocess.run(['colab','usage'],capture_output=True,text=True,check=True)
print(p.stdout,end='')
b=float(re.search(r'Current balance:\s*([0-9.]+)',p.stdout)[1])
base=json.loads((r/'usage_baseline.json').read_text())['before_session_balance_CU']
rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'balance_CU':b,'shared_account_decrease_CU':round(base-b,2),'raw':p.stdout}
with (r/'usage_history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
if base-b>=5.8:
 subprocess.run(['colab','stop','-s','rag-hotpotqa'],check=True)
 raise SystemExit('Conservative account-decrease safety guard reached; rag-hotpotqa stopped.')
