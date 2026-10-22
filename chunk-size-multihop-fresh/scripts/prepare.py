"""Freeze selected IDs and exact versioned public context corpus; restart-safe."""
import json,random,hashlib,time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
P=Path(__file__).resolve().parents[1]/'results';P.mkdir(exist_ok=True)
if not (P/'data.json').exists():
 ds=load_dataset('cmriat/2wikimultihopqa',split='validation')
 ix=list(range(len(ds)));random.Random(1320132).shuffle(ix)
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B');docs={};qs=[]
 for i in ix[16:3616]:
  r=ds[i];m=r['metadata'];titleids={}
  for title,sents in zip(m['context']['title'],m['context']['content']):
   text=' '.join(sents);did=hashlib.sha256((title+'\0'+text).encode()).hexdigest()[:24];titleids[title]=did
   if did not in docs:
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);offs=enc['offset_mapping'];spans=[];at=0
    for s in sents:
     start=at;end=at+len(s);inds=[j for j,(a,b) in enumerate(offs) if b>start and a<end]
     spans.append([min(inds),max(inds)+1] if inds else [0,0]);at=end+1
    docs[did]={'id':did,'title':title,'text':text,'token_ids':enc['input_ids'],'sentence_spans':spans}
  supports=[]
  for title,sid in zip(m['supporting_facts']['title'],m['supporting_facts']['sent_id']):
   did=titleids[title];supports.append({'doc_id':did,'sentence_id':sid,'span':docs[did]['sentence_spans'][sid]})
  qs.append({'id':r['id'],'question':r['question'],'answers':r['golden_answers'],'type':m['type'],'supports':supports,'gold_docs':list(dict.fromkeys(s['doc_id'] for s in supports))})
 data={'questions':qs,'documents':list(docs.values()),'info':{'dataset':'cmriat/2wikimultihopqa','split':'validation','fingerprint':ds._fingerprint,'seed':1320132,'smoke_ids':[ds[i]['id'] for i in ix[:16]],'tokenizer':'Qwen/Qwen3-8B'}}
 (P/'data.json').write_text(json.dumps(data,ensure_ascii=False));print('DATA_FROZEN',len(qs),len(docs),hashlib.sha256((P/'data.json').read_bytes()).hexdigest(),flush=True)
