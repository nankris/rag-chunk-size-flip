# Experiment 132: Does the RAG chunk-size winner flip?

**Verdict: INCONCLUSIVE.** Headline number: **2.1 percentage points** answer token F1 for 1024 minus 128 tokens on 600 fresh HotpotQA questions.

This run did not resolve the preregistered ranking-flip claim at the chosen confidence and practical thresholds. The observed differences are useful measurements, but they do not establish a reliable universal reversal.

Timing: estimated 100 minutes; actual 95.2 minutes elapsed, including recovery/waiting; NVIDIA L4; at most 3.76 CU (shared-account upper bound).

## Results

| Dataset | Chunk tokens | Recall@5 % [95% CI] | MRR@5 % [95% CI] | Answer F1 % [95% CI] | Exact match % [95% CI] | Mean prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| SQuAD pilot | 128 | 99.3 [98.3, 100.0] | 97.1 [95.5, 98.6] | 71.1 [66.5, 75.5] | 58.3 [52.7, 63.7] | 733 |
| SQuAD pilot | 256 | 97.7 [96.0, 99.3] | 95.1 [92.8, 97.1] | 68.1 [63.5, 72.7] | 55.0 [49.3, 60.7] | 1,361 |
| SQuAD pilot | 512 | 97.0 [95.0, 98.7] | 93.6 [91.0, 96.0] | 66.4 [61.6, 71.0] | 53.0 [47.3, 58.7] | 2,600 |
| SQuAD pilot | 1024 | 93.3 [90.3, 96.0] | 90.9 [87.7, 93.8] | 64.2 [59.2, 69.0] | 51.7 [46.0, 57.3] | 5,009 |
| HotpotQA confirm | 128 | 79.4 [77.2, 81.6] | 91.9 [90.1, 93.6] | 46.2 [42.6, 49.8] | 33.2 [29.5, 37.0] | 575 |
| HotpotQA confirm | 256 | 79.7 [77.5, 81.8] | 91.9 [90.1, 93.7] | 48.3 [44.8, 51.9] | 34.3 [30.7, 38.3] | 714 |
| HotpotQA confirm | 512 | 79.5 [77.3, 81.7] | 92.0 [90.1, 93.7] | 48.2 [44.7, 51.8] | 34.5 [30.8, 38.3] | 733 |
| HotpotQA confirm | 1024 | 79.5 [77.3, 81.7] | 92.0 [90.1, 93.7] | 48.3 [44.7, 51.8] | 34.5 [30.8, 38.3] | 733 |


Descriptive four-size point-estimate leaders on confirmation: Recall@5 256; MRR@5 512, 1024; F1 256; exact match 512, 1024 tokens. These are not additional significance tests.

Scores are macro averages over questions. All score intervals use 10,000 paired question bootstrap draws, seed 132. The table is also rendered in `figures/table.png` for Substack.

## Fixed decision rule and confirmation

H1 predicted 128 > 1024 on both document Recall@5 and MRR@5. H2 predicted 1024 > 128 on token F1. PROVEN requires every simultaneous lower bound >0 and every point gain >=5 points; DISPROVEN requires any simultaneous upper bound <0; otherwise INCONCLUSIVE. The five-point threshold represents one more useful result per 20 questions. Exact match and judging are secondary.

| Required contrast | Difference, points | 95% paired CI | Familywise 95% CI |
|---|---:|---:|---:|
| recall: 128 - 1024 | -0.08 | [-1.00, 0.83] | [-1.17, 1.00] |
| mrr: 128 - 1024 | -0.03 | [-0.72, 0.65] | [-0.88, 0.79] |
| f1: 1024 - 128 | 2.10 | [0.07, 4.18] | [-0.34, 4.62] |


The pilot alone would be labelled DISPROVEN by the same directional rule: large-minus-small F1 was -6.90 points, simultaneous CI [-12.93, -0.86]. It did not establish the predicted answer benefit. The final verdict uses confirmation only.

On confirmation, the ordinary F1 interval barely excludes zero, but the simultaneous interval crosses it and the 2.10-point gain misses the fixed five-point practical threshold. The planned retrieval advantages are absent. This is why a nominally positive answer result does not prove the full claim.

Pilot paired standard deviations were 0.238 for Recall, 0.241 for MRR, and 0.436 for F1. At the already-fixed confirmation n=600, these imply approximate simultaneous half-widths of 2.3, 2.4, and 4.3 percentage points if paired variability transferred across datasets. They are precision diagnostics, not guaranteed power on another dataset.

Bonferroni 98.333% individual intervals provide at least 95% simultaneous coverage over the three planned directional contrasts. No hypothesis or sample size changed after pilot outcomes. Preregistration commit: `554321b`.

## Method and diagnostics

Pilot: 300 SQuAD questions, 48 title-grouped articles. Confirmation: 600 HotpotQA distractor validation questions and 5,936 title-deduplicated context documents, searched globally. Seed 132 fixes selection. Chunking uses nonoverlapping Qwen-tokenizer windows; embeddings use Qwen3-Embedding-0.6B with normalized last-token pooling and full chunk text. FAISS performs exact inner-product search. Qwen3-4B answers greedily from top five chunks, thinking off, at most 64 output tokens. Actual served-model input/output counts are in every prediction record. Model revisions and package versions are saved.

The confirmation corpus has one duplicate-title text conflict, resolved by taking the first version before retrieval. Document relevance is a coarse label: a chunk from a gold document can score as relevant without containing the answer. HotpotQA Recall counts both supporting documents; MRR rewards the first one. These definitions can disagree with useful evidence retrieval.

At 512 versus 1024 tokens, identical top-five document lists occur on 98.0% of confirmation questions. At 512 tokens, 99.7% of documents fit a single chunk. Mean chunk lengths are 133.5 and 133.9 tokens respectively. A configured maximum is not a guarantee that documents fill it.

Qwen judged 60 questions per dataset at all four sizes, 240 candidate answers per dataset. Confirmation valid labels: 240/240. Agreement with F1 >=0.5: 84.2%; agreement with exact match: 75.4%. This uses the same model family and reference answers, not human adjudication. Disagreements are saved in `results/judge_disagreements.json`.

| Chunk tokens | Qwen judged accuracy % [95% CI], n=60 |
|---|---:|
| 128 | 48.3 [36.7, 61.7] |
| 256 | 50.0 [36.7, 63.3] |
| 512 | 48.3 [35.0, 61.7] |
| 1024 | 48.3 [35.0, 61.7] |


## Caveats and reproducibility

This is an independent test inspired by [Aqeel Abbas's article](https://medium.com/illumination/which-rag-chunk-size-is-best-the-answer-flips-depending-on-how-you-score-it-2c3ef424e50c); only its preview was accessible. We cannot claim to reproduce its hidden methods. Fixed top-five retrieval changes the reading budget across sizes, and HotpotQA snippets are often shorter than the largest chunk limits. This is not a full Wikipedia retrieval benchmark. The intervals condition on one corpus and model, ignore document clustering, and do not quantify model-seed or training uncertainty. SQuAD has only 48 articles, making its question-level intervals particularly optimistic for cross-document generalization. Public benchmarks may have appeared in pretraining. F1 can penalize valid paraphrases; the sampled model judge does not eliminate that issue.

All 3,600 answers and 480 judge calls were checkpointed. `scripts/validate.py` independently recomputes retrieval labels and checks completeness; `scripts/analyze.py`, `scripts/export.py`, and `scripts/publication.py` reproduce statistics, charts, tables, and this report. Resume via `scripts/sync.sh` while the session exists. Raw data, prompts' ingredients, predictions, scores, token counts, configs, and runtime logs are retained. See `results/usage_final.json` for honest shared-account CU accounting and `results/timing.json` for timing limitations.
