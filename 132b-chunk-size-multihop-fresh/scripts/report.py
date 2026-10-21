"""Render final numerical report from saved results. Blog is edited separately."""
import json
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P/'results';S=json.loads((R/'summary.json').read_text());h=S['headline'];T=json.loads((R/'timing.json').read_text());U=json.loads((R/'usage_final.json').read_text())
ci=lambda r,m:f'{r[m]*100:.2f} [{r[m+"_lo"]*100:.2f}, {r[m+"_hi"]*100:.2f}]'
rows=[]
for r in S['arms']:
 rows.append(f'| {"1,024-token budget" if r["regime"]=="budget" else "Top 5"} | {r["size"]} | {ci(r,"f1")} | {ci(r,"em")} | {ci(r,"support_recall")} | {r["prompt_tokens"]:.1f} | {r["k"]:.2f} |')
ct=[]
for r in S['contrasts']:ct.append(f'| {r["regime"]} | {r["size"]} − 128 | {r["gain_pp"]:+.2f} | [{r["simultaneous_lo_pp"]:+.2f}, {r["simultaneous_hi_pp"]:+.2f}] |')
a=S['arms'][0];b=S['arms'][3];eq=S['contrasts'][5];delta=h['gain_pp'];direction='improved' if delta>=0 else 'reduced'
plain=f'With the same evidence-token budget, increasing the chunk limit from 128 to 1,024 {direction} answer F1 by {abs(delta):.2f} percentage points on these fresh multi-hop questions. '
plain+=('That clears our prespecified success rule, but the result applies to this corpus and reader, not every RAG system.' if S['verdict']=='PROVEN' else 'The run did not establish the prespecified practical benefit, so larger chunks are not a justified default from this test alone.')
timing=f'Timing: estimated {T["estimated_minutes"]} minutes; actual {T["actual_minutes"]:.1f} minutes; NVIDIA L4; ≤{U["cu_upper_bound"]:.2f} CU (conservative usage bound; balance changes are shared and may lag).'
text=f'''# Experiment 132b: bigger RAG chunks on fresh multi-hop data

**Verdict: {S['verdict']}.** ONE headline number: **{delta:+.2f} percentage points answer F1**, 1,024 minus 128 chunk tokens under a fixed 1,024-token evidence budget. Simultaneous bootstrap CI: **[{h['simultaneous_lo_pp']:+.2f}, {h['simultaneous_hi_pp']:+.2f}] pp**.

{plain}

{timing}

## Main results

3,600 fresh 2WikiMultiHopQA validation questions in every arm, 28,800 answer records. All scores are macro averages over questions. Percentages and marginal 95% question-bootstrap CIs; prompt tokens include template, passage labels and question. Evidence budget excludes that overhead.

| Reading policy | Chunk limit | F1 % [95% CI] | EM % [95% CI] | Supporting-fact recall % [95% CI] | Mean input tokens | Mean k |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Table image: `figures/table.png`. Curves: `figures/answer_f1.png`. Paired contrasts and actual input lengths: `figures/effects_and_tokens.png`. Pipeline: `figures/diagram.png`, editable Excalidraw JSON alongside it.

## Fixed decision and simultaneous comparisons

Preregistration commit: `c9907fa`, before confirmation retrieval or answers. Primary H1 required the budget-controlled 1,024-minus-128 F1 simultaneous lower bound >0 AND observed gain >=1.5 pp. DISPROVEN if its simultaneous upper bound <1.5 pp; otherwise INCONCLUSIVE. These are the requested operational criteria, not a claim that a positive CI proves a population benefit >=1.5 pp. The title's approximate two points never replaced the registered 1.5-point threshold. Secondary arms cannot rescue the primary hypothesis. The unrounded primary gain was {delta:.6f} pp, narrowly below 1.5; its interval excludes zero but spans the practical margin. This explains the INCONCLUSIVE verdict without rounding the estimate into success.

| Policy | Contrast | Gain, pp | Simultaneous CI, pp |
|---|---|---:|---:|
'''+ '\n'.join(ct)+f'''

20,000 paired question-bootstrap draws, seed 1320132. Each contrast uses a Bonferroni 99.1667% two-sided percentile interval, targeting approximately 95% familywise coverage across all six planned F1 contrasts. Ordinary 95% intervals are also retained in `results/contrasts.csv`; they do not determine the verdict. Secondary metric CIs are marginal, not simultaneous.

## Why this sample size

Historical #132 HotpotQA answers gave paired SD 0.258396 and mean +2.10268 pp on n=600. The registered 1.5-point effect, six-comparison critical value and 80% sign-detection power imply n=3,594; we used 3,600. This power calculation assumes transfer of historical paired variance, and does not promise 80% chance of also clearing the practical point-estimate cutoff at its exact boundary. Observed primary paired SD was {h['paired_sd']:.6f}. The historical run stays INCONCLUSIVE; this fresh confirmation does not relabel it.

The first 16 shuffled 2Wiki questions were reserved and excluded; no reader pilot was run on them and no confirmatory outcomes were used for tuning. The historical HotpotQA run supplies pilot evidence from a different dataset and reader. Confirmation uses HF `cmriat/2wikimultihopqa`, validation, seed 1320132, IDs frozen before retrieval. All question contexts form one global searchable corpus of 19,945 paragraph versions. Exact title-and-text deduplication preserves conflicting paragraph versions and sentence labels.

## Method and budget control

Nonoverlapping Qwen3 tokenizer windows at 128/256/512/1024 tokens; Qwen3-Embedding-0.6B normalized last-token vectors; FAISS exact inner-product search. The main policy greedily consumes the ranked list to 1,024 original-document tokens and truncates only the last chunk. k therefore varies by query and size. The secondary policy takes five complete chunks. Neither regime is oracle retrieval, and the answer model is never given supporting labels. Titles are metadata and are not prepended to chunk text, matching the earlier text-only chunking recipe. Retrieval is one-shot dense search, with no iterative query decomposition or reranker. Text is decoded from token windows; actual input counts come from the served reader.

Reader: Qwen/Qwen3-8B via vLLM, FP16, NVIDIA L4, temperature zero, thinking disabled, max 64 answer tokens, seed 1320132. Same short-answer prompt in all arms. Full identical prompts share a cached deterministic output, while all arm-level records are retained. This reduces duplicate inference without substituting a different context or model. Token counts: {S['tokens']['served_unique_prompts']:,} unique prompts actually served, {S['tokens']['served_input']:,} input and {S['tokens']['served_output']:,} output tokens; logical totals across all answer records {S['tokens']['logical_input']:,} input and {S['tokens']['logical_output']:,} output tokens. Length finishes: {S['tokens']['max_token_finishes']} records.

Supporting-fact recall requires every token in the gold sentence to be covered by the union of retrieved spans from its exact document version. Document recall is also saved, but does not stand in for sentence evidence. F1/EM use provided answer aliases and normalized token overlap, including special yes/no handling. It is answer accuracy, not a test of whether the model actually reasoned over every hop.

Equal-k endpoint gain: {eq['gain_pp']:+.2f} pp, simultaneous CI [{eq['simultaneous_lo_pp']:+.2f}, {eq['simultaneous_hi_pp']:+.2f}]. Compare this with the main regime before attributing a gain to chunk size alone. Equal token counts still change chunk boundaries, passage count and prompt overhead, so this is a controlled policy comparison rather than a pure causal isolation of one scalar.

## Caveats and evidence trail

Most documents are short: median 72 tokens; only 410 exceed 512 and 14 exceed 1,024. A maximum chunk limit is not the actual passage length. This is a corpus assembled from benchmark contexts, not all Wikipedia or production documents. One public dataset, one reader/model seed, one query embedding recipe and one evidence budget limit generalization. Question-level bootstrap conditions on the corpus, treats sampled questions as independent and ignores shared documents, topics and templates. Public-data pretraining contamination is possible. Alias-based F1 can miss valid paraphrases, and no human evaluation was performed. Fact recall checks evidence availability, not faithfulness of the final answer.

The source [Aqeel Abbas article](https://medium.com/illumination/which-rag-chunk-size-is-best-the-answer-flips-depending-on-how-you-score-it-2c3ef424e50c) was available only as a public preview; we cannot claim to reproduce its hidden setup. Dataset documentation: [HF mirror](https://huggingface.co/datasets/cmriat/2wikimultihopqa) and [original 2Wiki repository](https://github.com/Alab-NII/2wikimultihop).

All raw answer and retrieval JSONL files, frozen data, model revisions, installed packages, corpus audit and logs are under `results/`. `scripts/validate.py` verifies completeness, evidence budgets, independent supporting-sentence coverage and scoring; `scripts/analyze.py` rebuilds statistics; `scripts/figures.py` rebuilds chart/table PNGs. `figures/colab_run.txt` contains GPU identification, stage timings, recovery evidence and inference batch logs. A Transformers metadata-field incompatibility was fixed before inference without changing the protocol. See `scripts/RESUME.md` for reboot recovery. Colab `b9-132b` was stopped after results were copied locally. No push was performed.
'''
(P/'writeup.md').write_text(text)
print('WROTE writeup.md')

readme=P/'README.md'
original=readme.read_text().split('\n## Completed result\n')[0]
result=f"\n## Completed result\n\n**{S['verdict']}**: budget-controlled 1024 minus 128 F1 = **{delta:+.2f} pp**, simultaneous CI **[{h['simultaneous_lo_pp']:+.2f}, {h['simultaneous_hi_pp']:+.2f}] pp**. The original prediction and decision rule above are preserved.\n\nAll 3,600 questions and 28,800 arm records passed validation. Qwen3-8B served {S['tokens']['served_unique_prompts']:,} unique prompts; exact token totals are in results/summary.json. Named Colab session b9-132b is stopped.\n\nSee [writeup.md](writeup.md) for numerical results and caveats, [blog.md](blog.md) for the publication draft, and figures/ for PNGs, editable diagram and terminal evidence. Local scripts reproduce the analysis without a GPU.\n"
readme.write_text(original+result)
