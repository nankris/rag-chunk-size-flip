"""Exact dense retrieval with resumable embedding batches and sentence-span recall."""
import os,json,time
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import numpy as np,torch,faiss
from transformers import AutoTokenizer,AutoModel
P=Path(__file__).resolve().parents[1]/'results';D=json.loads((P/'data.json').read_text());qs=D['questions'];docs=D['documents']
docmap={d['id']:d for d in docs}
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B',padding_side='left');ctok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
model=AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B',dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
(P/'embedding_config.json').write_text(json.dumps({'model':'Qwen/Qwen3-Embedding-0.6B','revision':__import__('huggingface_hub').model_info('Qwen/Qwen3-Embedding-0.6B').sha,'pooling':'last token','normalization':'L2','dtype':'float16','index':'FAISS IndexFlatIP','chunk_tokenizer':'Qwen/Qwen3-8B'},indent=2))
def load_complete(path,n):
 if not path.exists():return None
 try:
  a=np.load(path)
  if a.ndim!=2 or len(a)!=n:raise ValueError('Incomplete embedding shape')
  return a
 except (ValueError,EOFError,OSError):
  path.unlink();print('REBUILD_INCOMPLETE_ARRAY',path.name,flush=True);return None
def save_atomic(path,a):
 tmp=path.with_name(path.name+'.tmp.npy');np.save(tmp,a);os.replace(tmp,path)
def embed(texts,name):
 path=P/(name+'.npy')
 existing=load_complete(path,len(texts))
 if existing is not None:return existing
 order=None;cache_name=name
 if name=='embeddings_1024':
  order=np.argsort([len(ctok.encode(t,add_special_tokens=False)) for t in texts],kind='stable')
  texts=[texts[int(i)] for i in order];cache_name=name+'_lengthsorted'
 parts=[]
 for i in range(0,len(texts),48):
  f=P/f'{cache_name}.part{i}.npy'
  a=load_complete(f,len(texts[i:i+48]))
  if a is None:
   inp=tok(texts[i:i+48],padding=True,truncation=False,return_tensors='pt').to('cuda')
   with torch.inference_mode():
    a=torch.nn.functional.normalize(model(**inp).last_hidden_state[:,-1,:].float(),dim=1).cpu().numpy()
   save_atomic(f,a)
  parts.append(a)
  if i%960==0:print('EMBED',name,i,len(texts),flush=True)
 a=np.concatenate(parts)
 if order is not None:
  restored=np.empty_like(a);restored[order]=a;a=restored
 save_atomic(path,a)
 for f in P.glob(cache_name+'.part*.npy'):f.unlink()
 return a
qe=embed(['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'+q['question'] for q in qs],'query_embeddings')
def recall(q,sel):
 spans={}
 for c in sel:spans.setdefault(c['doc_id'],[]).append([c['start_token'],c['end_token']])
 found=0
 for s in q['supports']:
  lo,hi=s['span'];end=lo
  for a,b in sorted(spans.get(s['doc_id'],[])):
   if a<=end:end=max(end,b)
  found+=end>=hi
 return found/len(q['supports']),len(set(q['gold_docs'])&set(spans))/len(q['gold_docs'])
for size in [128,256,512,1024]:
 cp=P/f'chunks_{size}.json';rp=P/f'retrieval_{size}.jsonl'
 if rp.exists() and len(rp.read_text().splitlines())==len(qs)*2:continue
 t=time.time()
 if cp.exists():chunks=json.loads(cp.read_text())
 else:
  chunks=[]
  for d in docs:
   ids=d['token_ids']
   for st in range(0,len(ids),size):
    chunks.append({'doc_id':d['id'],'start_token':st,'end_token':min(st+size,len(ids)),'text':ctok.decode(ids[st:st+size]),'n_tokens':len(ids[st:st+size])})
  cp.write_text(json.dumps(chunks,ensure_ascii=False))
 z=embed([c['text'] for c in chunks],f'embeddings_{size}');index=faiss.IndexFlatIP(z.shape[1]);index.add(z);sc,ix=index.search(qe,128)
 with rp.open('w') as f:
  for q,inds,scores in zip(qs,ix,sc):
   for regime in ['budget','equal_k']:
    selected=[];used=0
    for j in inds:
     c=chunks[int(j)].copy();c['chunk_index']=int(j)
     if regime=='budget':
      remain=1024-used
      if c['n_tokens']>remain:
       c['end_token']=c['start_token']+remain;c['n_tokens']=remain
       # Reconstruct from original document token IDs to preserve exact token span.
       doc=docmap[c['doc_id']]
       c['text']=ctok.decode(doc['token_ids'][c['start_token']:c['end_token']])
     selected.append(c);used+=c['n_tokens']
     if (regime=='budget' and used>=1024) or (regime=='equal_k' and len(selected)==5):break
    rec,dr=recall(q,selected)
    row={'id':q['id'],'size':size,'regime':regime,'k':len(selected),'evidence_tokens':used,'support_recall':rec,'document_recall':dr,'selected':selected}
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print('RETRIEVAL_DONE',size,'chunks',len(chunks),'seconds',round(time.time()-t,2),flush=True)
print('RETRIEVAL_ALL_DONE',flush=True)
