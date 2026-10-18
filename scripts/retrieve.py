"""Resumable exact dense retrieval, full-length embeddings and FAISS."""
import os,time,json,gc,sys
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,numpy as np,faiss
from pathlib import Path
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
model_id='Qwen/Qwen3-Embedding-0.6B'
tok=AutoTokenizer.from_pretrained(model_id,padding_side='left')
chunk_tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
model=AutoModel.from_pretrained(model_id,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
(OUT/'embedding_config.json').write_text(json.dumps({'model':model_id,'revision':model.config._commit_hash,'tokenizer_revision':tok.init_kwargs.get('_commit_hash'),'dtype':'float16','pooling':'last token','normalization':'L2','attention':'sdpa','chunk_tokenizer':'Qwen/Qwen3-4B'},indent=2))
def embed(texts,path):
    if path.exists():return np.load(path)
    # Cache each batch, resume GPU work after a VM/process failure.
    parts=[]
    for i in range(0,len(texts),24):
        part=path.with_name(path.stem+f'.part{i}.npy')
        if part.exists():z=np.load(part)
        else:
            batch=tok(texts[i:i+24],padding=True,truncation=False,return_tensors='pt').to('cuda')
            assert batch.input_ids.shape[1]<32768
            with torch.inference_mode():
                z=model(**batch).last_hidden_state[:,-1,:].float()
                z=torch.nn.functional.normalize(z,p=2,dim=1).cpu().numpy()
            np.save(part,z)
        parts.append(z)
        if i%480==0:print('EMBED',path.name,i,len(texts),flush=True)
    z=np.concatenate(parts);np.save(path,z)
    for p in path.parent.glob(path.stem+'.part*.npy'):p.unlink()
    return z
for stage in sys.argv[1:] or ['pilot','confirm']:
    data=json.loads((OUT/f'{stage}_data.json').read_text());questions=data['questions']
    qt=['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'+q['question'] for q in questions]
    qe=embed(qt,OUT/f'{stage}_query_embeddings.npy')
    for size in [128,256,512,1024]:
        rp=OUT/f'{stage}_retrieval_{size}.jsonl'
        if rp.exists() and len(rp.read_text().splitlines())==len(questions):continue
        started=time.time();cp=OUT/f'{stage}_chunks_{size}.json'
        if cp.exists():chunks=json.loads(cp.read_text())
        else:
            chunks=[]
            for d in data['documents']:
                ids=chunk_tok.encode(d['text'],add_special_tokens=False)
                for st in range(0,len(ids),size):
                    text=chunk_tok.decode(ids[st:st+size])
                    chunks.append({'doc_id':d['id'],'start_token':st,'n_tokens':len(ids[st:st+size]),'text':text})
            cp.write_text(json.dumps(chunks,ensure_ascii=False))
        # Title retained as metadata, not prefixed to text: fixed token chunk size.
        emb=embed([c['text'] for c in chunks],OUT/f'{stage}_embeddings_{size}.npy')
        index=faiss.IndexFlatIP(emb.shape[1]);index.add(emb);scores,ix=index.search(qe,5)
        with rp.open('w') as f:
            for q,inds,sc in zip(questions,ix,scores):
                top=[chunks[int(j)] for j in inds];gold=set(q['gold_docs']);found=gold&{c['doc_id'] for c in top}
                ranks=[i+1 for i,c in enumerate(top) if c['doc_id'] in gold]
                r={'id':q['id'],'size':size,'recall':len(found)/len(gold),'mrr':1/min(ranks) if ranks else 0,'chunk_indices':inds.tolist(),'scores':sc.tolist(),'retrieved_doc_ids':[c['doc_id'] for c in top],'retrieved_context_tokens':sum(c['n_tokens'] for c in top)}
                f.write(json.dumps(r)+'\n')
        print('RETRIEVAL_DONE',stage,size,'chunks',len(chunks),'seconds',time.time()-started,'recall',np.mean([json.loads(l)['recall'] for l in rp.read_text().splitlines()]),flush=True)
print('RETRIEVAL_ALL_DONE',flush=True)
