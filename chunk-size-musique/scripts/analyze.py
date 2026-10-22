"""Fixed paired bootstrap, audits, decision rule, and publication charts."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

SIZES = [128, 256, 512, 1024]
METRICS = ['recall', 'mrr', 'f1', 'em']
LABELS = {
    'recall': 'Recall (Doc Hit Rate)',
    'mrr': 'MRR (Doc Rank)',
    'f1': 'Answer Token F1',
    'em': 'Exact Match'
}

def records(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def interval(x, ix, alpha=0.05):
    b = x[ix].mean(axis=1)
    return [float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))]

def main():
    summary = {}
    audit = {}
    
    stages = ['pilot', 'confirm']
    for stage in stages:
        data_p = OUT / f'{stage}_data.json'
        if not data_p.exists():
            continue
        data = json.loads(data_p.read_text())
        ids = [q['id'] for q in data['questions']]
        n = len(ids)
        
        # Check all answers exist
        all_exist = all((OUT / f'{stage}_answers_{s}.jsonl').exists() for s in SIZES)
        if not all_exist:
            print(f"Skipping {stage}: not all answer files exist yet.", flush=True)
            continue
            
        rng = np.random.default_rng(132)
        ix = rng.integers(n, size=(10000, n))
        
        rows = {}
        for size in SIZES:
            rr = records(OUT / f'{stage}_answers_{size}.jsonl')
            rmap = {r['id']: r for r in rr}
            rows[size] = [rmap[i] for i in ids]
            
        summary[stage] = {'n': n, 'metrics': {}, 'contrasts': {}, 'judge': {}}
        
        # Compute metrics per size
        for size in SIZES:
            summary[stage]['metrics'][size] = {}
            for m in METRICS:
                x = np.array([r[m] for r in rows[size]])
                summary[stage]['metrics'][size][m] = {
                    'mean': float(x.mean()),
                    'ci95': interval(x, ix, 0.05)
                }
            for m in ['prompt_tokens', 'completion_tokens', 'retrieved_context_tokens']:
                x = np.array([r[m] for r in rows[size]])
                summary[stage]['metrics'][size][m] = {
                    'mean': float(x.mean()),
                    'total': int(x.sum()),
                    'min': int(x.min()),
                    'max': int(x.max())
                }
                
        # Primary directional contrasts
        # H1: 128 - 1024 > 0 on recall
        # H2: 128 - 1024 > 0 on mrr
        # H3: 1024 - 128 > 0 on f1
        contrasts_spec = [
            ('recall', 1, '128 - 1024'),
            ('mrr', 1, '128 - 1024'),
            ('f1', -1, '1024 - 128'),
            ('em', -1, '1024 - 128')
        ]
        for m, sign, direction in contrasts_spec:
            # sign = 1 means 128 - 1024; sign = -1 means 1024 - 128
            diff = sign * (np.array([r[m] for r in rows[128]]) - np.array([r[m] for r in rows[1024]]))
            summary[stage]['contrasts'][m] = {
                'direction': direction,
                'mean': float(diff.mean()),
                'ci95': interval(diff, ix, 0.05),
                'ci_simultaneous': interval(diff, ix, 0.05 / 3),  # Bonferroni for 3 primary hypotheses
                'paired_sd': float(diff.std(ddof=1)),
                'se': float(diff.std(ddof=1) / math.sqrt(n))
            }
            
        # Decision rule on confirmation
        cc = [summary[stage]['contrasts'][m] for m in ['recall', 'mrr', 'f1']]
        # Margin is 0.015 (1.5 percentage points)
        margin = 0.015
        is_proven = all(c['ci_simultaneous'][0] > 0 and c['mean'] >= margin for c in cc)
        is_disproven = any(c['ci_simultaneous'][1] < 0 for c in cc)
        verdict = 'PROVEN' if is_proven else ('DISPROVEN' if is_disproven else 'INCONCLUSIVE')
        summary[stage]['verdict'] = verdict
        
        # Judge summary
        jp = OUT / f'{stage}_judge.jsonl'
        if jp.exists():
            jj = records(jp)
            for size in SIZES:
                j = [r for r in jj if r['size'] == size]
                valid = [r for r in j if r['judge_correct'] is not None]
                if valid:
                    jix = np.random.default_rng(132).integers(len(valid), size=(10000, len(valid)))
                    summary[stage]['judge'][size] = {
                        'n': len(j),
                        'valid': len(valid),
                        'accuracy': float(np.mean([r['judge_correct'] for r in valid])),
                        'ci95': interval(np.array([r['judge_correct'] for r in valid]), jix, 0.05),
                        'agreement_f1_half': float(np.mean([r['judge_correct'] == int(r['f1'] >= 0.5) for r in valid]))
                    }
                    
        # Audit info
        audit[stage] = {
            'documents': len(data['documents']),
            'questions': n,
            'unique_question_ids': len(set(ids)),
            'sizes': {}
        }
        for size in SIZES:
            chunks = json.loads((OUT / f'{stage}_chunks_{size}.json').read_text())
            doc_chunk_counts = {}
            for ch in chunks:
                doc_chunk_counts[ch['doc_id']] = doc_chunk_counts.get(ch['doc_id'], 0) + 1
            counts = np.array([c['n_tokens'] for c in chunks])
            rs = rows[size]
            audit[stage]['sizes'][size] = {
                'chunks': len(chunks),
                'single_chunk_documents_fraction': float(np.mean([doc_chunk_counts.get(d['id'], 0) == 1 for d in data['documents']])),
                'mean_chunk_tokens': float(counts.mean()),
                'max_chunk_tokens': int(counts.max()),
                'unique_retrieved_documents_mean': float(np.mean([len(set(r['retrieved_doc_ids'])) for r in rs]))
            }
            
    # Write summary and audit
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    print("Saved summary.json and audit.json", flush=True)
    
    render_figures(summary)

def render_figures(summary):
    """Render saved statistics without recomputing or writing result files."""
    summary = {stage: {**values, "metrics": {int(size): metrics for size, metrics in values["metrics"].items()}} for stage, values in summary.items()}
    # Generate publication figures if confirmation is present
    if 'confirm' in summary:
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        colors = {'pilot': '#64748b', 'confirm': '#0284c7'}
        
        # 1. metrics.png: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, m in zip(axes.flat, METRICS):
            for stg in ['pilot', 'confirm']:
                if stg not in summary: continue
                vals = [summary[stg]['metrics'][s][m] for s in SIZES]
                y = np.array([v['mean'] for v in vals]) * 100
                lo = np.array([v['ci95'][0] for v in vals]) * 100
                hi = np.array([v['ci95'][1] for v in vals]) * 100
                offset = -0.04 if stg == 'pilot' else 0.04
                label_txt = f"{stg.title()} (n={summary[stg]['n']})"
                ax.errorbar(np.arange(4) + offset, y, yerr=[y - lo, hi - y],
                            marker='o', capsize=4, label=label_txt, color=colors[stg], lw=1.5)
            ax.set(title=LABELS[m], xlabel='Chunk size (tokens)', ylabel='Score (%)')
            ax.set_xticks(range(4), SIZES)
            ax.grid(axis='y', alpha=0.25)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle('MuSiQue Multi-Hop RAG: Performance across Chunk Sizes (Budget-Controlled)', fontsize=15, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(FIG / 'metrics.png')
        plt.close(fig)
        
        # 2. contrasts.png: Forrest plot
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ms = ['recall', 'mrr', 'f1']
        labels_contrasts = ['Recall: 128 minus 1024', 'MRR: 128 minus 1024', 'Answer F1: 1024 minus 128']
        y_pos = np.arange(3)
        for off, stg in [(-0.1, 'pilot'), (0.1, 'confirm')]:
            if stg not in summary: continue
            cc = [summary[stg]['contrasts'][m] for m in ms]
            x = np.array([c['mean'] for c in cc]) * 100
            lo = np.array([c['ci_simultaneous'][0] for c in cc]) * 100
            hi = np.array([c['ci_simultaneous'][1] for c in cc]) * 100
            ax.errorbar(x, y_pos + off, xerr=[x - lo, hi - x], fmt='o', capsize=5,
                        color=colors[stg], label=f"{stg.title()} (Simultaneous 95% CI)", lw=1.8)
        ax.axvline(0, color='black', lw=1)
        ax.axvline(1.5, color='#d97706', ls='--', label='1.5 pp Practical Margin')
        ax.set_yticks(y_pos, labels_contrasts)
        ax.invert_yaxis()
        ax.set_xlabel('Paired Difference (Percentage Points; Positive Supports Flip Hypothesis)')
        ax.set_title('Pre-Registered Contrasts with Simultaneous Familywise 95% CIs', fontsize=13, weight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.legend(fontsize=9, loc='best')
        fig.tight_layout()
        fig.savefig(FIG / 'contrasts.png')
        plt.close(fig)
        
        # 3. tokens_vs_f1.png
        fig, ax = plt.subplots(figsize=(9, 5))
        for stg in ['pilot', 'confirm']:
            if stg not in summary: continue
            x = [summary[stg]['metrics'][s]['prompt_tokens']['mean'] for s in SIZES]
            y = [summary[stg]['metrics'][s]['f1']['mean'] * 100 for s in SIZES]
            ax.plot(x, y, 'o-', color=colors[stg], label=f"{stg.title()} (n={summary[stg]['n']})", lw=1.8)
            for a, b, s in zip(x, y, SIZES):
                ax.annotate(f"{s} tok", (a, b), textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax.set(xlabel='Mean Prompt Tokens (Input to Qwen3-4B)', ylabel='Answer Token F1 (%)',
               title='Budget-Controlled Tradeoff: Prompt Length vs Generation Quality')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / 'tokens_vs_f1.png')
        plt.close(fig)
        
        # 4. table.png
        def fmt(v):
            return f"{v['mean']*100:.2f} [{v['ci95'][0]*100:.2f}, {v['ci95'][1]*100:.2f}]"
            
        table_rows = []
        for stg in ['pilot', 'confirm']:
            if stg not in summary: continue
            for s in SIZES:
                mm = summary[stg]['metrics'][s]
                table_rows.append([
                    'Pilot' if stg == 'pilot' else 'Confirm',
                    str(s),
                    fmt(mm['recall']),
                    fmt(mm['mrr']),
                    fmt(mm['f1']),
                    fmt(mm['em']),
                    f"{mm['prompt_tokens']['mean']:,.0f}"
                ])
                
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.axis('off')
        col_labels = [
            'Dataset', 'Chunk\nTokens',
            'Recall (Doc Hit)\n% [95% CI]', 'MRR (Doc Rank)\n% [95% CI]',
            'Answer F1\n% [95% CI]', 'Exact Match\n% [95% CI]',
            'Mean Prompt\nTokens'
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.10, 0.08, 0.18, 0.18, 0.18, 0.18, 0.10]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            if r == 0:
                cell.set_facecolor('#0f172a')
                cell.set_text_props(color='white', weight='bold')
                cell.set_height(0.12)
            else:
                cell.set_facecolor('#f0f9ff' if r > 4 else ('#f8fafc' if r % 2 else 'white'))
        ax.set_title('MuSiQue Multi-Hop RAG Across Chunk Sizes (Budget-Controlled)', fontsize=15, pad=15, weight='bold')
        fig.text(0.5, 0.04, 'MuSiQue Answerable | Qwen3-Embedding-0.6B + FAISS | Qwen3-4B Greedy Answers | 10k Paired Bootstrap Resamples', ha='center', fontsize=9, color='#475569')
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(FIG / 'table.png', bbox_inches='tight')
        plt.close(fig)
        
        # 5. diagram.png (matplotlib visual architecture diagram)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)
        
        # Boxes styling
        box_kwargs = dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#0284c7", lw=1.5)
        box_model = dict(boxstyle="round,pad=0.5", fc="#e0f2fe", ec="#0369a1", lw=2)
        box_res = dict(boxstyle="round,pad=0.5", fc="#fef3c7", ec="#d97706", lw=1.5)
        
        ax.text(2, 6, "MuSiQue Dataset\n2,417 Multi-Hop Questions\n17,629 Deduplicated Docs", ha="center", va="center", bbox=box_kwargs, fontsize=10)
        ax.text(6, 6, "4 Chunking Regimes\n128, 256, 512, 1024 tokens\nQwen3 Tokenizer", ha="center", va="center", bbox=box_kwargs, fontsize=10)
        ax.text(10, 6, "Qwen3-Embedding-0.6B\nFAISS Dense Retrieval\nBudget Control: 1024 Tokens", ha="center", va="center", bbox=box_model, fontsize=10)
        
        ax.annotate('', xy=(4.2, 6), xytext=(3.3, 6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
        ax.annotate('', xy=(8.2, 6), xytext=(7.5, 6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
        
        ax.text(6, 3.8, "Budget-Controlled Context Assembly\nRetrieve chunks until cumulative tokens = 1024\n(Prevents 8x context advantage for 1024 chunks)", ha="center", va="center", bbox=box_model, fontsize=10)
        ax.annotate('', xy=(6, 4.5), xytext=(10, 5.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
        
        ax.text(2, 1.8, "Retrieval Evaluation\nRecall & MRR\nOver Gold Paragraphs", ha="center", va="center", bbox=box_res, fontsize=10)
        ax.text(6, 1.8, "Qwen3-4B Generation\nGreedy Decoding, Temp=0\nToken F1 & Exact Match", ha="center", va="center", bbox=box_model, fontsize=10)
        ax.text(10, 1.8, "10k Paired Bootstrap\nFamilywise CIs (Bonferroni)\nDecision Rule: 1.5 pp Margin", ha="center", va="center", bbox=box_res, fontsize=10)
        
        ax.annotate('', xy=(2, 2.5), xytext=(4.5, 3.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
        ax.annotate('', xy=(6, 2.5), xytext=(6, 3.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
        ax.annotate('', xy=(8.5, 1.8), xytext=(7.5, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
        
        ax.set_title("MuSiQue Pipeline: Budget-Controlled Multi-Hop RAG Evaluation", fontsize=14, weight="bold", pad=15)
        fig.tight_layout()
        fig.savefig(FIG / 'diagram.png')
        plt.close(fig)
        
        print("Generated all publication figures and diagram.png", flush=True)

if __name__ == '__main__':
    if '--figures-only' in sys.argv:
        render_figures(json.loads((OUT / 'summary.json').read_text()))
    else:
        main()
