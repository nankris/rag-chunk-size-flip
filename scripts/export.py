"""Export publication CSVs and judge disagreements after a complete run."""
import json,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'results'
s=json.loads((R/'summary.json').read_text())
rows=[]
for stage in ['pilot','confirm']:
 for size,mm in s[stage]['metrics'].items():
  r={'stage':stage,'chunk_tokens':int(size),'n':s[stage]['n']}
  for m in ['recall','mrr','f1','em']:
   r[m]=mm[m]['mean'];r[m+'_ci_low'],r[m+'_ci_high']=mm[m]['ci95']
  r['mean_prompt_tokens']=mm['prompt_tokens']['mean'];rows.append(r)
with (R/'main_table.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
allrows=[];dis=[]
for stage in ['pilot','confirm']:
 for size in [128,256,512,1024]:
  for r in map(json.loads,(R/f'{stage}_answers_{size}.jsonl').read_text().splitlines()):
   allrows.append({'stage':stage,**{k:r[k] for k in ['id','size','recall','mrr','f1','em','prompt_tokens','completion_tokens','prediction']}})
 for r in map(json.loads,(R/f'{stage}_judge.jsonl').read_text().splitlines()):
  if r['judge_correct'] is not None and r['judge_correct']!=int(r['f1']>=.5):dis.append({'stage':stage,**r})
with (R/'per_question.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
(R/'judge_disagreements.json').write_text(json.dumps(dis,indent=2))
print('Exported',len(rows),'table rows,',len(allrows),'predictions,',len(dis),'judge/F1 disagreements')
