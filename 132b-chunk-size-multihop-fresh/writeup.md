# Experiment 132b: bigger RAG chunks on fresh multi-hop data

**Verdict: INCONCLUSIVE.** ONE headline number: **+1.49 percentage points answer F1**, 1,024 minus 128 chunk tokens under a fixed 1,024-token evidence budget. Simultaneous bootstrap CI: **[+0.19, +2.76] pp**.

With the same evidence-token budget, increasing the chunk limit from 128 to 1,024 improved answer F1 by 1.49 percentage points on these fresh multi-hop questions. The run did not establish the prespecified practical benefit, so larger chunks are not a justified default from this test alone.

Timing: estimated 110 minutes; actual 105.0 minutes; NVIDIA L4; ≤4.93 CU (conservative usage bound; balance changes are shared and may lag).

## Main results

3,600 fresh 2WikiMultiHopQA validation questions in every arm, 28,800 answer records. All scores are macro averages over questions. Percentages and marginal 95% question-bootstrap CIs; prompt tokens include template, passage labels and question. Evidence budget excludes that overhead.

| Reading policy | Chunk limit | F1 % [95% CI] | EM % [95% CI] | Supporting-fact recall % [95% CI] | Mean input tokens | Mean k |
|---|---:|---:|---:|---:|---:|---:|
| 1,024-token budget | 128 | 28.11 [26.70, 29.52] | 24.39 [22.97, 25.81] | 68.45 [67.59, 69.29] | 1159.1 | 12.22 |
| 1,024-token budget | 256 | 29.36 [27.93, 30.76] | 25.50 [24.08, 26.92] | 69.39 [68.57, 70.19] | 1139.2 | 9.38 |
| 1,024-token budget | 512 | 29.30 [27.87, 30.72] | 25.22 [23.81, 26.64] | 69.26 [68.44, 70.06] | 1134.2 | 8.67 |
| 1,024-token budget | 1024 | 29.61 [28.17, 31.05] | 25.47 [24.06, 26.89] | 69.28 [68.47, 70.09] | 1133.7 | 8.61 |
| Top 5 | 128 | 25.81 [24.44, 27.20] | 22.58 [21.22, 23.94] | 64.47 [63.62, 65.32] | 531.6 | 5.00 |
| Top 5 | 256 | 27.27 [25.85, 28.66] | 23.83 [22.42, 25.22] | 67.02 [66.20, 67.82] | 686.9 | 5.00 |
| Top 5 | 512 | 27.31 [25.90, 28.69] | 23.86 [22.47, 25.25] | 67.57 [66.76, 68.37] | 768.9 | 5.00 |
| Top 5 | 1024 | 27.54 [26.13, 28.92] | 24.06 [22.67, 25.44] | 67.78 [66.97, 68.58] | 786.4 | 5.00 |

Table image: `figures/table.png`. Curves: `figures/answer_f1.png`. Paired contrasts and actual input lengths: `figures/effects_and_tokens.png`. Pipeline: `figures/diagram.png`, editable Excalidraw JSON alongside it.

## Fixed decision and simultaneous comparisons

Preregistration commit: `c9907fa`, before confirmation retrieval or answers. Primary H1 required the budget-controlled 1,024-minus-128 F1 simultaneous lower bound >0 AND observed gain >=1.5 pp. DISPROVEN if its simultaneous upper bound <1.5 pp; otherwise INCONCLUSIVE. These are the requested operational criteria, not a claim that a positive CI proves a population benefit >=1.5 pp. The title's approximate two points never replaced the registered 1.5-point threshold. Secondary arms cannot rescue the primary hypothesis. The unrounded primary gain was 1.494898 pp, narrowly below 1.5; its interval excludes zero but spans the practical margin. This explains the INCONCLUSIVE verdict without rounding the estimate into success.

| Policy | Contrast | Gain, pp | Simultaneous CI, pp |
|---|---|---:|---:|
| budget | 256 − 128 | +1.24 | [+0.05, +2.46] |
| budget | 512 − 128 | +1.18 | [-0.07, +2.46] |
| budget | 1024 − 128 | +1.49 | [+0.19, +2.76] |
| equal_k | 256 − 128 | +1.46 | [+0.35, +2.56] |
| equal_k | 512 − 128 | +1.50 | [+0.40, +2.63] |
| equal_k | 1024 − 128 | +1.73 | [+0.62, +2.86] |

20,000 paired question-bootstrap draws, seed 1320132. Each contrast uses a Bonferroni 99.1667% two-sided percentile interval, targeting approximately 95% familywise coverage across all six planned F1 contrasts. Ordinary 95% intervals are also retained in `results/contrasts.csv`; they do not determine the verdict. Secondary metric CIs are marginal, not simultaneous.

## Why this sample size

Historical #132 HotpotQA answers gave paired SD 0.258396 and mean +2.10268 pp on n=600. The registered 1.5-point effect, six-comparison critical value and 80% sign-detection power imply n=3,594; we used 3,600. This power calculation assumes transfer of historical paired variance, and does not promise 80% chance of also clearing the practical point-estimate cutoff at its exact boundary. Observed primary paired SD was 0.290676. The historical run stays INCONCLUSIVE; this fresh confirmation does not relabel it.

The first 16 shuffled 2Wiki questions were reserved and excluded; no reader pilot was run on them and no confirmatory outcomes were used for tuning. The historical HotpotQA run supplies pilot evidence from a different dataset and reader. Confirmation uses HF `cmriat/2wikimultihopqa`, validation, seed 1320132, IDs frozen before retrieval. All question contexts form one global searchable corpus of 19,945 paragraph versions. Exact title-and-text deduplication preserves conflicting paragraph versions and sentence labels.

## Method and budget control

Nonoverlapping Qwen3 tokenizer windows at 128/256/512/1024 tokens; Qwen3-Embedding-0.6B normalized last-token vectors; FAISS exact inner-product search. The main policy greedily consumes the ranked list to 1,024 original-document tokens and truncates only the last chunk. k therefore varies by query and size. The secondary policy takes five complete chunks. Neither regime is oracle retrieval, and the answer model is never given supporting labels. Titles are metadata and are not prepended to chunk text, matching the earlier text-only chunking recipe. Retrieval is one-shot dense search, with no iterative query decomposition or reranker. Text is decoded from token windows; actual input counts come from the served reader.

Reader: Qwen/Qwen3-8B via vLLM, FP16, NVIDIA L4, temperature zero, thinking disabled, max 64 answer tokens, seed 1320132. Same short-answer prompt in all arms. Full identical prompts share a cached deterministic output, while all arm-level records are retained. This reduces duplicate inference without substituting a different context or model. Token counts: 20,740 unique prompts actually served, 19,773,591 input and 83,471 output tokens; logical totals across all answer records 26,423,626 input and 116,881 output tokens. Length finishes: 22 records.

Supporting-fact recall requires every token in the gold sentence to be covered by the union of retrieved spans from its exact document version. Document recall is also saved, but does not stand in for sentence evidence. F1/EM use provided answer aliases and normalized token overlap, including special yes/no handling. It is answer accuracy, not a test of whether the model actually reasoned over every hop.

Equal-k endpoint gain: +1.73 pp, simultaneous CI [+0.62, +2.86]. Compare this with the main regime before attributing a gain to chunk size alone. Equal token counts still change chunk boundaries, passage count and prompt overhead, so this is a controlled policy comparison rather than a pure causal isolation of one scalar.

## Caveats and evidence trail

Most documents are short: median 72 tokens; only 410 exceed 512 and 14 exceed 1,024. A maximum chunk limit is not the actual passage length. This is a corpus assembled from benchmark contexts, not all Wikipedia or production documents. One public dataset, one reader/model seed, one query embedding recipe and one evidence budget limit generalization. Question-level bootstrap conditions on the corpus, treats sampled questions as independent and ignores shared documents, topics and templates. Public-data pretraining contamination is possible. Alias-based F1 can miss valid paraphrases, and no human evaluation was performed. Fact recall checks evidence availability, not faithfulness of the final answer.

The source [Aqeel Abbas article](https://medium.com/illumination/which-rag-chunk-size-is-best-the-answer-flips-depending-on-how-you-score-it-2c3ef424e50c) was available only as a public preview; we cannot claim to reproduce its hidden setup. Dataset documentation: [HF mirror](https://huggingface.co/datasets/cmriat/2wikimultihopqa) and [original 2Wiki repository](https://github.com/Alab-NII/2wikimultihop).

All raw answer and retrieval JSONL files, frozen data, model revisions, installed packages, corpus audit and logs are under `results/`. `scripts/validate.py` verifies completeness, evidence budgets, independent supporting-sentence coverage and scoring; `scripts/analyze.py` rebuilds statistics; `scripts/figures.py` rebuilds chart/table PNGs. `figures/colab_run.txt` contains GPU identification, stage timings, recovery evidence and inference batch logs. A Transformers metadata-field incompatibility was fixed before inference without changing the protocol. See `scripts/RESUME.md` for reboot recovery. Colab `b9-132b` was stopped after results were copied locally. No push was performed.
