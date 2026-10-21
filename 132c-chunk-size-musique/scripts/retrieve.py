"""Resumable dense retrieval with full-length embeddings, FAISS exact search, and budget control."""
import os, time, json, gc, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import numpy as np
import faiss
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'

EMBED_MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'
TOKENIZER_ID = 'Qwen/Qwen3-4B'
BUDGET_TOKENS = 1024

print("Loading embedding model and tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID, padding_side='left')
chunk_tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
model = AutoModel.from_pretrained(EMBED_MODEL_ID, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()

(OUT / 'embedding_config.json').write_text(json.dumps({
    'embedding_model': EMBED_MODEL_ID,
    'tokenizer': TOKENIZER_ID,
    'dtype': 'float16',
    'pooling': 'last token',
    'normalization': 'L2',
    'budget_tokens': BUDGET_TOKENS
}, indent=2))

def embed(texts, path):
    if path.exists():
        print(f"Loading existing {path.name} ({len(texts)} embeddings)", flush=True)
        return np.load(path)
    
    parts = []
    batch_size = 48
    for i in range(0, len(texts), batch_size):
        part = path.with_name(path.stem + f'.part{i}.npy')
        if part.exists():
            z = np.load(part)
        else:
            batch = tok(texts[i:i + batch_size], padding=True, truncation=False, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                z = model(**batch).last_hidden_state[:, -1, :].float()
                z = torch.nn.functional.normalize(z, p=2, dim=1).cpu().numpy()
            np.save(part, z)
        parts.append(z)
        if i % 960 == 0 or i + batch_size >= len(texts):
            print(f"EMBED {path.name}: {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
            
    z = np.concatenate(parts, axis=0)
    np.save(path, z)
    for p in path.parent.glob(path.stem + '.part*.npy'):
        p.unlink()
    return z

def run_stage(stage):
    data = json.loads((OUT / f'{stage}_data.json').read_text())
    questions = data['questions']
    documents = data['documents']
    
    # 1. Embed queries
    query_texts = [
        f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q['question']}"
        for q in questions
    ]
    qe = embed(query_texts, OUT / f'{stage}_query_embeddings.npy')
    
    for size in [128, 256, 512, 1024]:
        retrieval_path = OUT / f'{stage}_retrieval_{size}.jsonl'
        if retrieval_path.exists():
            lines = retrieval_path.read_text().strip().splitlines()
            if len(lines) == len(questions):
                print(f"Skipping {stage} size {size}: already completed ({len(lines)} records)", flush=True)
                continue
                
        t0 = time.time()
        # 2. Chunk documents
        chunks_path = OUT / f'{stage}_chunks_{size}.json'
        if chunks_path.exists():
            chunks = json.loads(chunks_path.read_text())
        else:
            chunks = []
            for d in documents:
                ids = chunk_tok.encode(d['text'], add_special_tokens=False)
                for st in range(0, len(ids), size):
                    chunk_ids = ids[st:st + size]
                    text = chunk_tok.decode(chunk_ids)
                    chunks.append({
                        'doc_id': d['id'],
                        'start_token': st,
                        'n_tokens': len(chunk_ids),
                        'text': text
                    })
            chunks_path.write_text(json.dumps(chunks, ensure_ascii=False))
            print(f"Created {len(chunks)} chunks of size {size} for {stage}", flush=True)
            
        # 3. Embed chunks
        ce = embed([c['text'] for c in chunks], OUT / f'{stage}_chunk_embeddings_{size}.npy')
        
        # 4. FAISS index
        dim = ce.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(ce)
        
        # 5. Search
        top_k_search = min(50, len(chunks))
        D, I = index.search(qe, top_k_search)
        
        # 6. Build budget-controlled and fixed-k candidate sets
        records = []
        for q_idx, q in enumerate(questions):
            gold_docs = set(q['gold_docs'])
            retrieved_chunk_indices = I[q_idx].tolist()
            
            # Budget-controlled selection: pack chunks until cumulative tokens >= BUDGET_TOKENS
            budget_chunks = []
            budget_tokens_accum = 0
            for c_idx in retrieved_chunk_indices:
                c = chunks[c_idx]
                budget_chunks.append(c_idx)
                budget_tokens_accum += c['n_tokens']
                if budget_tokens_accum >= BUDGET_TOKENS:
                    break
                    
            budget_doc_ids = [chunks[idx]['doc_id'] for idx in budget_chunks]
            unique_budget_doc_ids = set(budget_doc_ids)
            
            # Recall: fraction of gold docs found in budget-controlled set
            recall = len(unique_budget_doc_ids & gold_docs) / len(gold_docs) if gold_docs else 0.0
            
            # MRR: reciprocal rank of first gold document chunk (1-indexed)
            mrr = 0.0
            for rank_0, c_idx in enumerate(retrieved_chunk_indices):
                if chunks[c_idx]['doc_id'] in gold_docs:
                    mrr = 1.0 / (rank_0 + 1)
                    break
                    
            # Secondary check: fixed top-5 chunks
            fixed5_chunks = retrieved_chunk_indices[:5]
            fixed5_doc_ids = set(chunks[idx]['doc_id'] for idx in fixed5_chunks)
            fixed5_recall = len(fixed5_doc_ids & gold_docs) / len(gold_docs) if gold_docs else 0.0
            fixed5_tokens = sum(chunks[idx]['n_tokens'] for idx in fixed5_chunks)
            
            records.append({
                'id': q['id'],
                'size': size,
                'budget_tokens_target': BUDGET_TOKENS,
                'retrieved_context_tokens': budget_tokens_accum,
                'chunk_indices': budget_chunks,
                'retrieved_doc_ids': budget_doc_ids,
                'recall': recall,
                'mrr': mrr,
                'fixed_k_recall': fixed5_recall,
                'fixed_k_tokens': fixed5_tokens
            })
            
        retrieval_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n')
        dur = time.time() - t0
        mean_recall = np.mean([r['recall'] for r in records])
        mean_mrr = np.mean([r['mrr'] for r in records])
        print(f"RETRIEVAL_DONE {stage} size {size}: {len(records)} items, {dur:.1f}s | Recall: {mean_recall:.4f}, MRR: {mean_mrr:.4f}", flush=True)

def main():
    stages = sys.argv[1:] or ['pilot', 'confirm']
    for stage in stages:
        print(f"=== Starting retrieval for stage: {stage} ===", flush=True)
        run_stage(stage)
    print("ALL RETRIEVAL COMPLETE", flush=True)

if __name__ == '__main__':
    main()
