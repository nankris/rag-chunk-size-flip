"""Batched, checkpointed Qwen3-4B answers with budget-controlled context and reference judging."""
import os, json, time, sys, re, string, collections
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'

def normalize_answer(s):
    s = s.lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def compute_f1_em(pred, answers):
    p = normalize_answer(pred).split()
    vals = []
    for a in answers:
        g = normalize_answer(a).split()
        common = sum((collections.Counter(p) & collections.Counter(g)).values())
        f1 = (2 * common / (len(p) + len(g))) if (p and g) else float(p == g)
        em = float(normalize_answer(pred) == normalize_answer(a))
        vals.append((f1, em))
    return max(v[0] for v in vals), max(v[1] for v in vals)

def main():
    print("Loading Qwen3-4B for generation...", flush=True)
    model_id = 'Qwen/Qwen3-4B'
    tok = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='cuda',
        attn_implementation='sdpa'
    ).eval()
    
    (OUT / 'generation_config.json').write_text(json.dumps({
        'model': model_id,
        'dtype': 'float16',
        'max_new_tokens': 64,
        'temperature': 0,
        'padding_side': 'left',
        'batch_size': 12,
        'seed': 132
    }, indent=2))
    
    stages = sys.argv[1:] or ['pilot', 'confirm']
    for stage in stages:
        data = json.loads((OUT / f'{stage}_data.json').read_text())
        qmap = {q['id']: q for q in data['questions']}
        
        for size in [128, 256, 512, 1024]:
            chunks = json.loads((OUT / f'{stage}_chunks_{size}.json').read_text())
            retrieval_records = [
                json.loads(l) for l in (OUT / f'{stage}_retrieval_{size}.jsonl').read_text().splitlines()
            ]
            ans_path = OUT / f'{stage}_answers_{size}.jsonl'
            
            done = set()
            if ans_path.exists():
                for line in ans_path.read_text().splitlines():
                    if line.strip():
                        try:
                            done.add(json.loads(line)['id'])
                        except Exception:
                            pass
                        
            todo = [r for r in retrieval_records if r['id'] not in done]
            print(f"Stage {stage} size {size}: {len(done)} already done, {len(todo)} remaining", flush=True)
            if not todo:
                continue
                
            batch_size = 12
            for st in range(0, len(todo), batch_size):
                batch = todo[st:st + batch_size]
                prompts = []
                for r in batch:
                    q = qmap[r['id']]
                    ctx = '\n\n'.join(
                        f"Passage {i+1}:\n{chunks[j]['text']}"
                        for i, j in enumerate(r['chunk_indices'])
                    )
                    messages = [
                        {
                            'role': 'system',
                            'content': 'Answer the question using the passages. Give only a short answer, without explanation. If the passages do not provide the answer, say unknown.'
                        },
                        {
                            'role': 'user',
                            'content': f"Passages:\n{ctx}\n\nQuestion: {q['question']}"
                        }
                    ]
                    prompt_str = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    prompts.append(prompt_str)
                    
                t0 = time.time()
                inputs = tok(prompts, return_tensors='pt', padding=True, truncation=False).to('cuda')
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id
                    )
                dur = time.time() - t0
                
                input_len = inputs.input_ids.shape[1]
                gen_tokens = outputs[:, input_len:]
                preds = tok.batch_decode(gen_tokens, skip_special_tokens=True)
                
                with ans_path.open('a') as f:
                    for r, pred, in_ids, out_ids in zip(batch, preds, inputs.input_ids, gen_tokens):
                        clean_pred = pred.strip()
                        gold_answers = qmap[r['id']]['answers']
                        f1, em = compute_f1_em(clean_pred, gold_answers)
                        
                        non_pad_prompt = int((in_ids != tok.pad_token_id).sum().item())
                        non_pad_completion = int(len(out_ids))
                        
                        rec = {
                            **r,
                            'prediction': clean_pred,
                            'gold_answers': gold_answers,
                            'f1': f1,
                            'em': em,
                            'prompt_tokens': non_pad_prompt,
                            'completion_tokens': non_pad_completion,
                            'finish_reason': 'stop' if non_pad_completion < 64 else 'length'
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    
                del inputs, outputs, gen_tokens
                torch.cuda.empty_cache()
                    
                if (st + len(batch)) % 120 == 0 or st + len(batch) >= len(todo):
                    print(f"ANSWER_BATCH {stage} size {size}: {st + len(batch)}/{len(todo)} in {dur:.2f}s ({len(batch)/dur:.1f} req/s)", flush=True)

        # Blinded-to-size reference judge for first 60 questions
        jp = OUT / f'{stage}_judge.jsonl'
        judge_done = set()
        if jp.exists():
            for line in jp.read_text().splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                        judge_done.add((rec['id'], rec['size']))
                    except Exception:
                        pass
                    
        judge_todo = []
        for size in [128, 256, 512, 1024]:
            ans_records = {
                json.loads(l)['id']: json.loads(l)
                for l in (OUT / f'{stage}_answers_{size}.jsonl').read_text().splitlines()
            }
            for q in data['questions'][:60]:
                if (q['id'], size) not in judge_done:
                    judge_todo.append((q, ans_records[q['id']]))
                    
        print(f"Stage {stage} judge: {len(judge_todo)} evaluations to run", flush=True)
        for st in range(0, len(judge_todo), 24):
            batch = judge_todo[st:st + 24]
            prompts = []
            for q, r in batch:
                messages = [
                    {
                        'role': 'system',
                        'content': 'Assess whether a candidate answer correctly answers a question, using the reference answer. Allow equivalent wording, but reject contradictions, unknown, or missing essential facts. Output exactly CORRECT or INCORRECT.'
                    },
                    {
                        'role': 'user',
                        'content': f"Question: {q['question']}\nReference answers: {json.dumps(q['answers'])}\nCandidate answer: {r['prediction']}"
                    }
                ]
                prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                
            inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            input_len = inputs.input_ids.shape[1]
            judge_texts = tok.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            
            with jp.open('a') as f:
                for (q, r), jt, in_ids, out_ids in zip(batch, judge_texts, inputs.input_ids, outputs[:, input_len:]):
                    s = jt.strip()
                    correct = 1 if s.startswith('CORRECT') else (0 if s.startswith('INCORRECT') else None)
                    rec = {
                        'id': q['id'],
                        'size': r['size'],
                        'judge_text': s,
                        'judge_correct': correct,
                        'f1': r['f1'],
                        'em': r['em'],
                        'prediction': r['prediction'],
                        'gold_answers': q['answers'],
                        'prompt_tokens': int((in_ids != tok.pad_token_id).sum().item()),
                        'completion_tokens': int(len(out_ids))
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            del inputs, outputs
            torch.cuda.empty_cache()
            print(f"JUDGE_BATCH {stage}: {st + len(batch)}/{len(judge_todo)}", flush=True)

    print("ALL GENERATION AND JUDGING COMPLETE", flush=True)

if __name__ == '__main__':
    main()
