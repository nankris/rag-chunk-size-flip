"""Freeze public MuSiQue datasets and question IDs before retrieval."""
import json, random, hashlib, time
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def save(name, docs, qs, info):
    p = OUT / f'{name}_data.json'
    p.write_text(json.dumps({'documents': docs, 'questions': qs, 'info': info}, ensure_ascii=False))
    print(name, 'documents:', len(docs), 'questions:', len(qs), 'sha256:', hashlib.sha256(p.read_bytes()).hexdigest(), flush=True)

def main():
    # 1. Pilot dataset: 300 questions from train split
    if not (OUT / 'pilot_data.json').exists():
        print("Loading MuSiQue train for pilot...", flush=True)
        ds_train = load_dataset('dgslibisey/MuSiQue', split='train')
        ix = list(range(len(ds_train)))
        random.Random(132).shuffle(ix)
        pilot_ix = ix[:300]
        
        docs_map = {}
        qs = []
        for i in pilot_ix:
            r = ds_train[i]
            # Group paragraphs by title
            gold_titles = []
            for p in r['paragraphs']:
                title = p['title']
                text = p['paragraph_text']
                docs_map.setdefault(title, []).append(text)
                if p['is_supporting']:
                    gold_titles.append(title)
            answers = [r['answer']] + [a for a in r.get('answer_aliases', []) if a != r['answer']]
            qs.append({
                'id': r['id'],
                'question': r['question'],
                'answers': answers,
                'gold_docs': list(dict.fromkeys(gold_titles))
            })
        
        # Build deduplicated document corpus
        docs = []
        for title, texts in docs_map.items():
            dedup_texts = list(dict.fromkeys(texts))
            docs.append({'id': title, 'text': title + '\n\n' + '\n\n'.join(dedup_texts)})
        
        save('pilot', docs, qs, {
            'dataset': 'dgslibisey/MuSiQue',
            'split': 'train',
            'sample_size': len(qs),
            'seed': 132
        })

    # 2. Confirmation dataset: all 2,417 questions from validation split
    if not (OUT / 'confirm_data.json').exists():
        print("Loading MuSiQue validation for confirmation...", flush=True)
        ds_val = load_dataset('dgslibisey/MuSiQue', split='validation')
        
        docs_map = {}
        qs = []
        for r in ds_val:
            gold_titles = []
            for p in r['paragraphs']:
                title = p['title']
                text = p['paragraph_text']
                docs_map.setdefault(title, []).append(text)
                if p['is_supporting']:
                    gold_titles.append(title)
            answers = [r['answer']] + [a for a in r.get('answer_aliases', []) if a != r['answer']]
            qs.append({
                'id': r['id'],
                'question': r['question'],
                'answers': answers,
                'gold_docs': list(dict.fromkeys(gold_titles))
            })
            
        docs = []
        for title, texts in docs_map.items():
            dedup_texts = list(dict.fromkeys(texts))
            docs.append({'id': title, 'text': title + '\n\n' + '\n\n'.join(dedup_texts)})
            
        save('confirm', docs, qs, {
            'dataset': 'dgslibisey/MuSiQue',
            'split': 'validation',
            'sample_size': len(qs),
            'seed': 132
        })

if __name__ == '__main__':
    main()
