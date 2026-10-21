"""Record conservative CU accounting and stop only the named experiment session."""
import subprocess,re,json,datetime,math
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P/'results'
assert (R/'DONE').exists() or (R/'CAP_STOP').exists(), 'Copy completed results or explicit cap-failure evidence before stopping'
if (R/'DONE').exists():
 rows=[json.loads(l) for l in (R/'answers.jsonl').read_text().splitlines()]
 assert len(rows)==28800 and len({(r['id'],r['regime'],r['size']) for r in rows})==28800,'Take a complete final snapshot before stopping'
 for size in [128,256,512,1024]:assert len((R/f'retrieval_{size}.jsonl').read_text().splitlines())==7200
u=subprocess.run(['colab','usage'],capture_output=True,text=True,check=True).stdout
balance=float(re.search(r'Current balance:\s*([0-9.]+)',u).group(1));start=datetime.datetime.fromisoformat(json.loads((R/'timing.json').read_text())['start_ist'])
r=subprocess.run(['colab','stop','-s','b9-132b'],capture_output=True,text=True,check=True);(R/'stop_session.txt').write_text(r.stdout+r.stderr);end=datetime.datetime.now(datetime.timezone.utc);elapsed=(end-start).total_seconds()/3600
out={'baseline_balance':127.05,'final_balance':balance,'account_delta_cu':round(127.05-balance,4),'account_usage_text':u,'session':'b9-132b','GPU':'NVIDIA L4','inferred_session_rate_cu_per_hour':1.54,'lease_hours_upper_bound':elapsed,'rate_time_cu_upper_bound':elapsed*1.54,'cu_upper_bound':math.ceil(max(127.05-balance,elapsed*1.54)*100)/100,'stop_utc':end.isoformat(),'note':'Conservative bound is max(shared-account observed decrement, 1.54 CU/h times total experiment elapsed until stop). Experiment start precedes allocation, so rate-time bound includes extra setup time. Shared balance changes may lag and are not exact session billing.'}
(R/'usage_final.json').write_text(json.dumps(out,indent=2));(R/'SESSION_STOPPED').write_text(end.isoformat());print(json.dumps(out,indent=2))
