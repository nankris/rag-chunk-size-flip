"""Preserve total elapsed time and label partially observed stage timing honestly."""
import datetime,json,re
from pathlib import Path
r=Path(__file__).resolve().parents[1];o=r/'results';p=o/'timing.json';z=json.loads(p.read_text())
now=datetime.datetime.now(datetime.timezone.utc);parse=datetime.datetime.fromisoformat
z['end_ist']=now.astimezone(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat()
z['actual_minutes']=(now-parse(z['start_ist'])).total_seconds()/60
z['GPU']='NVIDIA L4'
log=(r/'figures/colab_run.txt').read_text();st=re.findall(r'STAGE_START pilot (\S+)',log)[-1];en=re.findall(r'STAGE_END confirm (\S+)',log)[-1]
prior=(o/'prior_run.log').read_text();seconds=sum(float(s) for s in re.findall(r'RETRIEVAL_DONE pilot (?:128|256|512) chunks \d+ seconds ([0-9.]+)',prior))
resumed=(parse(en)-parse(st)).total_seconds()/60
local=json.loads((o/'local_stages.json').read_text())
z['stage_minutes']={'setup':(parse(z['setup_scripts_ready_ist'])-parse(z['start_ist'])).total_seconds()/60,'GPU_run':resumed+seconds/60,'eval':(parse(local['evaluation_end_utc'])-parse(local['evaluation_start_utc'])).total_seconds()/60,'writeup':(now-parse(local['publication_preparation_start_utc'])).total_seconds()/60}
z['stage_timing_notes']={'GPU_run':'Lower bound: complete resumed pilot+confirmation pipeline wall span plus original completed retrieval timings. Exact end of vanished original GPU session is unavailable, so lost startup/idle time cannot be allocated accurately. This is pipeline elapsed time, not a GPU kernel profiler.','setup':'Time until initial scripts-ready marker; excludes subsequent recovery and installation.','writeup':'Elapsed since publication preparation started, overlapping GPU run and evaluation. Stages are not additive.','total':'Exact wall time since original experiment start, including interruptions, slot waiting, recovery, GPU pipeline, evaluation and publication preparation.'}
z['observed_resumed_GPU_pipeline_minutes']=resumed;z['original_completed_retrieval_minutes']=seconds/60
z['CU_account_decrease_upper_bound']=json.loads((o/'usage_final.json').read_text())['conservative_account_decrease_CU']
p.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
