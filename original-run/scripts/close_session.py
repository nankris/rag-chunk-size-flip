"""Release only the finished named session and preserve honest usage evidence."""
import subprocess,json,re,datetime
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'results'
assert (r/'DONE').exists(), 'Copy all finished results before closing the session.'
def usage():
 p=subprocess.run(['colab','usage'],capture_output=True,text=True,check=True)
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw':p.stdout,'balance_CU':float(re.search(r'Current balance:\s*([0-9.]+)',p.stdout)[1])}
before=usage()
p=subprocess.run(['colab','stop','-s','rag-hotpotqa'],capture_output=True,text=True)
(r/'stop_session.txt').write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(p.stdout+p.stderr)
after=usage();base=json.loads((r/'usage_baseline.json').read_text())['before_session_balance_CU']
z={'GPU':'NVIDIA L4','session':'rag-hotpotqa','initial_balance_CU':base,'before_stop':before,'after_stop':after,'experiment_CU_exact':None,'conservative_account_decrease_CU':round(base-after['balance_CU'],2),'limit_CU':6,'session_stopped':True,'accounting_note':'Exact per-session charges are not exposed by colab usage. The observed total account decrease includes other experiments and allocation waiting, so it is reported as a conservative upper bound on this experiment, not an exact measured session charge.'}
(r/'usage_final.json').write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
