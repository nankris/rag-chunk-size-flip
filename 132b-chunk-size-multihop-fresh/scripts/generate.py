"""Checkpointed vLLM reader. Identical full prompts share deterministic generation."""
import os,json,time,hashlib,re,string,collections
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'results'
def norm(s):
 s=''.join(c for c in s.lower() if c not in string.punctuation);return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def score(pred,answers):
 p=norm(pred);scores=[]
 for a in answers:
  g=norm(a);pt=p.split();gt=g.split();n=sum((collections.Counter(pt)&collections.Counter(gt)).values())
  f=2*n/(len(pt)+len(gt)) if pt and gt else float(pt==gt)
  if (p in ['yes','no','noanswer'] or g in ['yes','no','noanswer']) and p!=g:f=0
  scores.append((f,float(p==g)))
 return max(v[0] for v in scores),max(v[1] for v in scores)
def repair_tail(path):
 if not path.exists():return
 lines=path.read_bytes().splitlines(keepends=True);end=0
 for i,line in enumerate(lines):
  try:json.loads(line)
  except (json.JSONDecodeError,UnicodeDecodeError):
   if i!=len(lines)-1:raise RuntimeError('Non-tail JSONL corruption: '+str(path))
   with path.open('r+b') as f:f.truncate(end)
   print('REPAIRED_INCOMPLETE_TAIL',str(path),flush=True);return
  end+=len(line)
 if lines and not lines[-1].endswith(b'\n'):
  with path.open('ab') as f:f.write(b'\n')
def main():
 from transformers import AutoTokenizer,AutoConfig
 from vllm import LLM,SamplingParams
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B');qs={q['id']:q for q in json.loads((P/'data.json').read_text())['questions']}
 llm=LLM(model='Qwen/Qwen3-8B',dtype='half',max_model_len=6500,gpu_memory_utilization=.91,max_num_seqs=32,enable_prefix_caching=True,enforce_eager=True,seed=1320132)
 (P/'generation_config.json').write_text(json.dumps({'model':'Qwen/Qwen3-8B','revision':__import__('huggingface_hub').model_info('Qwen/Qwen3-8B').sha,'dtype':'float16','temperature':0,'max_tokens':64,'thinking':False,'seed':1320132,'max_model_len':6500},indent=2))
 path=P/'answers.jsonl';repair_tail(path);cache={};done=set()
 if path.exists():
  for l in path.read_text().splitlines():
   r=json.loads(l);done.add((r['id'],r['size'],r['regime']));cache[r['prompt_hash']]={k:r[k] for k in ['prediction','prompt_tokens','completion_tokens','finish_reason']}
 rows=[]
 for size in [128,1024,256,512]:rows.extend(map(json.loads,(P/f'retrieval_{size}.jsonl').read_text().splitlines()))
 rows=[r for r in rows if (r['id'],r['size'],r['regime']) not in done]
 for st in range(0,len(rows),64):
  batch=rows[st:st+64];prompts={};hashes=[]
  for r in batch:
   ctx='\n\n'.join(f'Passage {i+1}: '+c['text'] for i,c in enumerate(r['selected']))
   prompt=tok.apply_chat_template([{'role':'system','content':'Answer the question using the passages. Give only a short answer, without explanation. If the passages do not provide the answer, say unknown.'},{'role':'user','content':f'Passages:\n{ctx}\n\nQuestion: {qs[r["id"]]["question"]}'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   h=hashlib.sha256(prompt.encode()).hexdigest();hashes.append(h)
   if h not in cache:prompts[h]=prompt
  start=time.time()
  if prompts:
   outs=llm.generate(list(prompts.values()),SamplingParams(temperature=0,max_tokens=64),use_tqdm=False)
   for h,o in zip(prompts,outs):cache[h]={'prediction':o.outputs[0].text.strip(),'prompt_tokens':len(o.prompt_token_ids),'completion_tokens':len(o.outputs[0].token_ids),'finish_reason':o.outputs[0].finish_reason}
  with path.open('a') as f:
   for r,h in zip(batch,hashes):
    c=cache[h];f1,em=score(c['prediction'],qs[r['id']]['answers']);record={k:v for k,v in r.items() if k!='selected'}
    record.update(c);record.update({'f1':f1,'em':em,'prompt_hash':h,'gold_answers':qs[r['id']]['answers'],'reused_identical_prompt':h not in prompts})
    f.write(json.dumps(record,ensure_ascii=False)+'\n')
   f.flush();os.fsync(f.fileno())
  print('ANSWER_BATCH',st+len(batch),'of',len(rows),'unique_generated',len(prompts),'seconds',round(time.time()-start,2),flush=True)
 (P/'DONE').write_text(str(time.time()));print('GENERATION_ALL_DONE',flush=True)
if __name__=='__main__':main()
