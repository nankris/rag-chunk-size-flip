"""Independent integrity checks of saved spans, budget and answer accounting."""
import json,hashlib,math,collections,re
from pathlib import Path
from generate import score
P=Path(__file__).resolve().parents[1]/'results';D=json.loads((P/'data.json').read_text());qs={q['id']:q for q in D['questions']};docs={d['id']:d for d in D['documents']};ret={};count=0
for size in [128,256,512,1024]:
 for l in (P/f'retrieval_{size}.jsonl').read_text().splitlines():
  r=json.loads(l);selected=r['selected'];assert len(selected)==r['k'];assert all(0<c['n_tokens']<=size for c in selected)
  assert sum(c['n_tokens'] for c in selected)==r['evidence_tokens']
  if r['regime']=='budget':assert r['evidence_tokens']==1024
  else:assert len(selected)==5
  covered=0
  for s in qs[r['id']]['supports']:
   required=set(range(*s['span']));actual=set()
   for c in selected:
    assert c['end_token']-c['start_token']==c['n_tokens'];assert c['end_token']<=len(docs[c['doc_id']]['token_ids'])
    if c['doc_id']==s['doc_id']:actual.update(range(c['start_token'],c['end_token']))
   covered+=required<=actual
  assert math.isclose(covered/len(qs[r['id']]['supports']),r['support_recall'])
  ret[(r['id'],size,r['regime'])]=r;count+=1
answers=[json.loads(l) for l in (P/'answers.jsonl').read_text().splitlines()];seen=set();hashes={}
for r in answers:
 key=(r['id'],r['size'],r['regime']);assert key not in seen;seen.add(key);assert key in ret
 f,e=score(r['prediction'],qs[r['id']]['answers']);assert abs(f-r['f1'])<1e-12 and e==r['em']
 for k in ['support_recall','document_recall','evidence_tokens','k']:assert r[k]==ret[key][k]
 assert r['completion_tokens']<=64 and r['prompt_tokens']<6500
 h=r['prompt_hash'];payload=(r['prediction'],r['prompt_tokens'],r['completion_tokens'])
 assert h not in hashes or hashes[h]==payload;hashes[h]=payload
assert len(answers)==len(ret)==3600*8
logged_unique=sum(map(int,re.findall(r'ANSWER_BATCH \d+ of \d+ unique_generated (\d+)',(P/'run.log').read_text())))
assert logged_unique==len(hashes),(logged_unique,len(hashes))
assert len(hashes)==json.loads((P/'prompt_identity_audit.json').read_text())['distinct_context_questions']
out={'passed':True,'questions':len(qs),'documents':len(docs),'retrieval_rows':count,'answer_rows':len(answers),'unique_prompts':len(hashes),'logged_unique_generations':logged_unique,'data_sha256':hashlib.sha256((P/'data.json').read_bytes()).hexdigest(),'checks':['complete Cartesian product','no smoke overlap (analysis)','exact evidence budgets','fully covered gold sentence token spans','independent span-union recall','scores recomputed','reader token limits','identical prompt cache consistency','task generation log counts match distinct full prompts and independent retrieval identity audit']}
(P/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
