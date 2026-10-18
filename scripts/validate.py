"""Audit saved experiment artifacts and independently recompute retrieval metrics."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 report={}
 for stage,n in [('pilot',300),('confirm',600)]:
  data=json.loads((OUT/f'{stage}_data.json').read_text());qs={q['id']:q for q in data['questions']};docs={d['id'] for d in data['documents']};assert len(qs)==n
  assert all(set(q['gold_docs'])<=docs for q in qs.values())
  for size in [128,256,512,1024]:
   chunk=json.loads((OUT/f'{stage}_chunks_{size}.json').read_text());ans=rows(OUT/f'{stage}_answers_{size}.jsonl');retr=rows(OUT/f'{stage}_retrieval_{size}.jsonl');assert len(ans)==len(retr)==n
   assert {r['id'] for r in ans}==set(qs)
   for r in ans:
    cs=[chunk[i] for i in r['chunk_indices']];gold=set(qs[r['id']]['gold_docs']);top=[c['doc_id'] for c in cs];assert len(cs)==5
    assert r['recall']==len(gold&set(top))/len(gold)
    rank=next((i+1 for i,d in enumerate(top) if d in gold),None)
    assert r['mrr']==(1/rank if rank else 0)
    assert all(0<=r[k]<=1 for k in ['recall','mrr','f1','em'])
    assert r['completion_tokens']<=64 and r['prompt_tokens']+r['completion_tokens']<=6500
   report[f'{stage}_{size}']={'n':len(ans),'unique_ids':len({r['id'] for r in ans}),'retrieval_recomputed':True,'lengths_valid':True}
  jj=rows(OUT/f'{stage}_judge.jsonl');assert len(jj)==240 and len({(r['id'],r['size']) for r in jj})==240
  assert {r['id'] for r in jj}=={q['id'] for q in data['questions'][:60]}
  report[f'{stage}_judge']={'n':len(jj),'valid_labels':sum(r['judge_correct'] is not None for r in jj)}
 assert not ({q['id'] for q in json.loads((OUT/'pilot_data.json').read_text())['questions']}&{q['id'] for q in json.loads((OUT/'confirm_data.json').read_text())['questions']})
 (OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
