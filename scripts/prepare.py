"""Freeze public datasets and question IDs before retrieval. Run on Colab."""
import json,random,hashlib,time
from pathlib import Path
from datasets import load_dataset
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
def save(name,docs,qs,info):
    p=OUT/(name+'_data.json')
    p.write_text(json.dumps({'documents':docs,'questions':qs,'info':info},ensure_ascii=False))
    print(name,'documents',len(docs),'questions',len(qs),'sha256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
def main():
    if not (OUT/'pilot_data.json').exists():
        ds=load_dataset('rajpurkar/squad',split='validation')
        docs={}; by_context={}
        for r in ds:
            docs.setdefault(r['title'],{})[r['context']]=None
            by_context.setdefault(r['context'],r)
        docs=[{'id':t,'text':'\n\n'.join(ps)} for t,ps in docs.items()]
        candidates=list(by_context.values());random.Random(132).shuffle(candidates)
        qs=[{'id':r['id'],'question':r['question'],'answers':r['answers']['text'],'gold_docs':[r['title']]} for r in candidates[:300]]
        save('pilot',docs,qs,{'dataset':'rajpurkar/squad','split':'validation','fingerprint':ds._fingerprint})
    if not (OUT/'confirm_data.json').exists():
        ds=load_dataset('hotpotqa/hotpot_qa','distractor',split='validation')
        ix=list(range(len(ds)));random.Random(132).shuffle(ix);qs=[];docs={};conflicts=0
        for i in ix[:600]:
            r=ds[i]
            for title,sents in zip(r['context']['title'],r['context']['sentences']):
                txt=''.join(sents)
                if title in docs and docs[title]!=txt:conflicts+=1
                # First occurrence is the fixed corpus version, including conflicts.
                docs.setdefault(title,txt)
            qs.append({'id':r['id'],'question':r['question'],'answers':[r['answer']],'gold_docs':list(dict.fromkeys(r['supporting_facts']['title']))})
        save('confirm',[{'id':k,'text':v} for k,v in docs.items()],qs,{'dataset':'hotpotqa/hotpot_qa','config':'distractor','split':'validation','fingerprint':ds._fingerprint,'duplicate_title_text_conflicts':conflicts})
if __name__=='__main__': main()
