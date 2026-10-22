"""Reproduce preregistered paired CIs and descriptive diagnostics."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1];R=P/'results'
data=json.loads((R/'data.json').read_text());ids=[q['id'] for q in data['questions']]
assert len(ids)==3600 and len(set(ids))==3600
assert not set(ids)&set(data['info']['smoke_ids'])
rows=[json.loads(l) for l in (R/'answers.jsonl').read_text().splitlines()]
assert len(rows)==28800
keys=[(r['id'],r['regime'],r['size']) for r in rows];assert len(set(keys))==len(keys)
lookup={k:r for k,r in zip(keys,rows)}
arms=[(regime,s) for regime in ['budget','equal_k'] for s in [128,256,512,1024]]
metrics=['f1','em','support_recall','document_recall','prompt_tokens','completion_tokens','evidence_tokens','k']
a=np.array([[[lookup[(qid,reg,s)][m] for m in metrics] for reg,s in arms] for qid in ids])
rng=np.random.default_rng(1320132);boot=[]
for i in range(0,20000,100):
 ix=rng.integers(0,len(ids),size=(100,len(ids)));boot.append(a[ix].mean(axis=1))
b=np.concatenate(boot);means=a.mean(axis=0);table=[]
for j,(reg,s) in enumerate(arms):
 r={'regime':reg,'size':s,'n':len(ids)}
 for k,m in enumerate(metrics):
  r[m]=means[j,k];r[m+'_lo'],r[m+'_hi']=np.quantile(b[:,j,k],[.025,.975]).tolist()
 table.append(r)
contrasts=[]
for base in [0,4]:
 for j in [base+1,base+2,base+3]:
  ds=b[:,j,0]-b[:,base,0];lo,hi=np.quantile(ds,[.05/12,1-.05/12]);gain=means[j,0]-means[base,0]
  contrasts.append({'regime':arms[j][0],'size':arms[j][1],'gain_pp':100*gain,'simultaneous_lo_pp':100*lo,'simultaneous_hi_pp':100*hi,'ordinary_lo_pp':100*np.quantile(ds,.025),'ordinary_hi_pp':100*np.quantile(ds,.975),'paired_sd':float((a[:,j,0]-a[:,base,0]).std(ddof=1))})
h=contrasts[2];verdict='PROVEN' if h['simultaneous_lo_pp']>0 and h['gain_pp']>=1.5 else 'DISPROVEN' if h['simultaneous_hi_pp']<1.5 else 'INCONCLUSIVE'
diagnostics={}
for size in [128,256,512,1024]:
 cs=json.loads((R/f'chunks_{size}.json').read_text());lens=[c['n_tokens'] for c in cs];counts={}
 for c in cs:counts[c['doc_id']]=counts.get(c['doc_id'],0)+1
 diagnostics[str(size)]={'n_chunks':len(cs),'mean_chunk_tokens':float(np.mean(lens)),'median_chunk_tokens':float(np.median(lens)),'one_chunk_document_fraction':sum(n==1 for n in counts.values())/len(counts)}
unique={r['prompt_hash']:r for r in rows}
tokens={'logical_input':sum(r['prompt_tokens'] for r in rows),'logical_output':sum(r['completion_tokens'] for r in rows),'served_unique_prompts':len(unique),'served_input':sum(r['prompt_tokens'] for r in unique.values()),'served_output':sum(r['completion_tokens'] for r in unique.values()),'total_answers':len(rows),'max_token_finishes':sum(r['finish_reason']=='length' for r in rows)}
summary={'verdict':verdict,'headline':h,'arms':table,'contrasts':contrasts,'chunk_diagnostics':diagnostics,'tokens':tokens,'bootstrap_draws':20000,'bootstrap_seed':1320132,'preregistration_commit':'c9907fa'}
(R/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(table).to_csv(R/'main_table.csv',index=False);pd.DataFrame(rows).to_csv(R/'per_question.csv',index=False);pd.DataFrame(contrasts).to_csv(R/'contrasts.csv',index=False)
print(json.dumps({'verdict':verdict,'headline':h,'tokens':tokens,'chunks':diagnostics},indent=2))
