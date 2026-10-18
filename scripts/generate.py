"""Batched, checkpointed Qwen answers plus blinded-to-size reference judging."""
import os,json,time,sys,re,string,collections
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
def norm(s):
    s=s.lower();s=''.join(c for c in s if c not in string.punctuation);s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())
def score(pred,answers):
    p=norm(pred).split();vals=[]
    for a in answers:
        g=norm(a).split(); common=sum((collections.Counter(p)&collections.Counter(g)).values())
        f=2*common/(len(p)+len(g)) if p and g else float(p==g)
        # Standard HotpotQA treatment of yes/no/noanswer.
        if (norm(a) in ['yes','no','noanswer'] or norm(pred) in ['yes','no','noanswer']) and norm(a)!=norm(pred):f=0
        vals.append((f,float(norm(pred)==norm(a))))
    return max(v[0] for v in vals),max(v[1] for v in vals)
def main():
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
    llm=LLM(model='Qwen/Qwen3-4B',dtype='half',max_model_len=6500,gpu_memory_utilization=.88,max_num_seqs=32,enable_prefix_caching=True,enforce_eager=True,seed=132)
    (OUT/'generation_config.json').write_text(json.dumps({'model':'Qwen/Qwen3-4B','revision':getattr(llm.llm_engine.model_config.hf_config,'_commit_hash',None),'dtype':'float16','max_new_tokens':64,'temperature':0,'seed':132,'thinking':False,'max_model_len':6500},indent=2))
    def prompt(msg):return tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    stages=sys.argv[1:] or ['pilot','confirm']
    for stage in stages:
        data=json.loads((OUT/f'{stage}_data.json').read_text());qmap={q['id']:q for q in data['questions']}
        for size in [128,256,512,1024]:
            chunks=json.loads((OUT/f'{stage}_chunks_{size}.json').read_text());retr=[json.loads(l) for l in (OUT/f'{stage}_retrieval_{size}.jsonl').read_text().splitlines()]
            path=OUT/f'{stage}_answers_{size}.jsonl';done={json.loads(l)['id'] for l in path.read_text().splitlines()} if path.exists() else set()
            todo=[r for r in retr if r['id'] not in done]
            for st in range(0,len(todo),32):
                batch=todo[st:st+32];prompts=[]
                for r in batch:
                    q=qmap[r['id']];ctx='\n\n'.join(f'Passage {i+1}: '+chunks[j]['text'] for i,j in enumerate(r['chunk_indices']))
                    prompts.append(prompt([{'role':'system','content':'Answer the question using the passages. Give only a short answer, without explanation. If the passages do not provide the answer, say unknown.'},{'role':'user','content':f'Passages:\n{ctx}\n\nQuestion: {q["question"]}'}]))
                start=time.time();outs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=64),use_tqdm=False)
                with path.open('a') as f:
                    for r,o in zip(batch,outs):
                        pred=o.outputs[0].text.strip();f1,em=score(pred,qmap[r['id']]['answers'])
                        rec={**r,'prediction':pred,'gold_answers':qmap[r['id']]['answers'],'f1':f1,'em':em,'prompt_tokens':len(o.prompt_token_ids),'completion_tokens':len(o.outputs[0].token_ids),'finish_reason':o.outputs[0].finish_reason}
                        f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                print('ANSWER_BATCH',stage,size,st+len(batch),'of',len(todo),'seconds',round(time.time()-start,2),flush=True)
        # All four answers for the first 60 deterministically shuffled questions.
        jp=OUT/f'{stage}_judge.jsonl';done={(r['id'],r['size']) for r in map(json.loads,jp.read_text().splitlines())} if jp.exists() else set();todo=[]
        for size in [128,256,512,1024]:
            amap={r['id']:r for r in map(json.loads,(OUT/f'{stage}_answers_{size}.jsonl').read_text().splitlines())}
            for q in data['questions'][:60]:
                if (q['id'],size) not in done:todo.append((q,amap[q['id']]))
        for st in range(0,len(todo),32):
            batch=todo[st:st+32];prompts=[]
            for q,r in batch:
                prompts.append(prompt([{'role':'system','content':'Assess whether a candidate answer correctly answers a question, using the reference answer. Allow equivalent wording, but reject contradictions, unknown, or missing essential facts. Output exactly CORRECT or INCORRECT.'},{'role':'user','content':f'Question: {q["question"]}\nReference answers: {json.dumps(q["answers"])}\nCandidate answer: {r["prediction"]}'}]))
            outs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=8),use_tqdm=False)
            with jp.open('a') as f:
                for (q,r),o in zip(batch,outs):
                    s=o.outputs[0].text.strip();f.write(json.dumps({'id':q['id'],'size':r['size'],'judge_text':s,'judge_correct':1 if s.startswith('CORRECT') else 0 if s.startswith('INCORRECT') else None,'f1':r['f1'],'em':r['em'],'prediction':r['prediction'],'gold_answers':q['answers'],'prompt_tokens':len(o.prompt_token_ids),'completion_tokens':len(o.outputs[0].token_ids)})+'\n')
                f.flush();os.fsync(f.fileno())
            print('JUDGE_BATCH',stage,st+len(batch),'of',len(todo),flush=True)
    print('GENERATION_ALL_DONE',flush=True)
if __name__=='__main__':main()
