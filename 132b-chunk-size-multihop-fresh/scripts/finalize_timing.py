"""Finalize a nonoverlapping wall-clock partition after reviewing the artifacts."""
import json,datetime,re,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P/'results';utc=datetime.timezone.utc;now=datetime.datetime.now(utc)
marks=R/'stage_marks.json'
if '--eval-end' in sys.argv:
 j=json.loads(marks.read_text()) if marks.exists() else {};j['eval_end_utc']=now.isoformat();marks.write_text(json.dumps(j,indent=2));raise SystemExit(0)
j=json.loads((R/'timing.json').read_text());start=datetime.datetime.fromisoformat(j['start_ist']);text=(R/'run.log').read_text();first=re.search(r'STAGE_START retrieve ([^\s]+)',text);gpu_start=datetime.datetime.fromisoformat(first.group(1));gpu_end=datetime.datetime.fromtimestamp(float((R/'DONE').read_text()),utc);eval_end=datetime.datetime.fromisoformat(json.loads(marks.read_text())['eval_end_utc']);usage=json.loads((R/'usage_final.json').read_text())
j.update({'end_ist':now.astimezone(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'actual_minutes':(now-start).total_seconds()/60,'GPU':'NVIDIA L4','CU_upper_bound':usage['cu_upper_bound'],'stage_minutes':{'setup':(gpu_start-start).total_seconds()/60,'GPU run':(gpu_end-gpu_start).total_seconds()/60,'eval':(eval_end-gpu_end).total_seconds()/60,'writeup':(now-eval_end).total_seconds()/60},'GPU_run_definition':'Wall interval from first retrieval attempt through DONE, including reader download/warmup and one checkpoint-preserving operational restart. GPU-run time is not a pure kernel benchmark.','overlap_note':'Writing preparation overlapped GPU work; the four reported stages partition elapsed wall time rather than summing labor time.'})
(R/'timing.json').write_text(json.dumps(j,indent=2));print(json.dumps(j,indent=2))
p=P/'README.md';s=p.read_text();s=re.sub(r'Timing: estimated 110 minutes \(setup 20, GPU run 65, analysis 10, writing 15\); actual pending\. GPU: NVIDIA L4 planned; CU pending\.',f'Timing: estimated 110 minutes (setup 20, GPU run 65, analysis 10, writing 15); actual {j["actual_minutes"]:.1f} minutes. GPU: NVIDIA L4; CU ≤{usage["cu_upper_bound"]:.2f} (conservative accounting bound).',s)
# Update an already-finalized timing line on a second call.
s=re.sub(r'actual [0-9.]+ minutes\. GPU: NVIDIA L4; CU ≤[0-9.]+ \(conservative accounting bound\)\.',f'actual {j["actual_minutes"]:.1f} minutes. GPU: NVIDIA L4; CU ≤{usage["cu_upper_bound"]:.2f} (conservative accounting bound).',s)
p.write_text(s)
