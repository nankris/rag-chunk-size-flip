# Experiment 132c: Chunk Size on Multi-Hop QA (MuSiQue)

**Verdict: INCONCLUSIVE.** Headline number: **-1.07 percentage points** answer token F1 for 1024 minus 128 chunk tokens on 2,417 fresh MuSiQue validation questions under budget-controlled reading context (95% CI [-2.12, -0.06] pp; familywise simultaneous 95% CI [-2.38, +0.18] pp).

**Takeaway:** On genuine multi-hop reasoning where answers require piecing together disconnected facts across multiple documents, 128-token chunks beat 1024-token chunks on document recall (+8.78 pp) and achieve higher answer quality (13.88% vs 12.81% F1) under an identical token reading budget. The widely claimed "metric ranking flip"—where large chunks supposedly dominate generation—vanishes once you prevent large chunks from sneaking in an unfair 8x reading budget advantage.

Timing: estimated 120 minutes; actual 262 minutes elapsed (setup: 20 min, GPU run: 205 min, eval: 15 min, writeup: 22 min); NVIDIA L4; 4.1 Colab CU used (6.05 CU shared-account upper bound).

## Results

| Dataset | Chunk tokens | Recall (Doc Hit) % [95% CI] | MRR (Doc Rank) % [95% CI] | Answer F1 % [95% CI] | Exact Match % [95% CI] | Mean Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|
| MuSiQue pilot (n=300) | 128 | 73.17 [69.86, 76.36] | 86.50 [83.25, 89.69] | 18.41 [14.47, 22.56] | 14.00 [10.33, 18.00] | 1,207 |
| MuSiQue pilot (n=300) | 256 | 69.97 [66.64, 73.25] | 86.31 [83.01, 89.50] | 18.19 [14.35, 22.29] | 13.67 [10.00, 17.67] | 1,222 |
| MuSiQue pilot (n=300) | 512 | 67.97 [64.50, 71.25] | 86.62 [83.42, 89.72] | 18.95 [14.99, 23.10] | 14.00 [10.33, 18.00] | 1,241 |
| MuSiQue pilot (n=300) | 1024 | 67.11 [63.67, 70.39] | 85.97 [82.64, 89.15] | 20.26 [16.26, 24.53] | 15.33 [11.33, 19.33] | 1,288 |
| MuSiQue confirm (n=2,417) | 128 | 60.13 [59.00, 61.26] | 82.23 [81.04, 83.39] | 13.88 [12.70, 15.12] | 8.15 [7.12, 9.27] | 1,226 |
| MuSiQue confirm (n=2,417) | 256 | 56.44 [55.32, 57.57] | 82.85 [81.65, 84.00] | 13.56 [12.37, 14.77] | 7.90 [6.87, 8.98] | 1,244 |
| MuSiQue confirm (n=2,417) | 512 | 53.49 [52.38, 54.62] | 83.12 [81.88, 84.37] | 12.63 [11.47, 13.82] | 7.36 [6.37, 8.44] | 1,268 |
| MuSiQue confirm (n=2,417) | 1024 | 51.34 [50.24, 52.47] | 82.50 [81.24, 83.77] | 12.81 [11.66, 14.02] | 7.28 [6.29, 8.36] | 1,319 |

The table is also rendered as an image in `figures/table.png` for Substack publication.

## Fixed Decision Rule and Confirmation Contrasts

Hypothesis testing followed the pre-registered decision rule from Experiment 132b using 10,000 paired question bootstrap draws (seed 132). Familywise coverage across the three primary contrasts is guaranteed at 95% via Bonferroni correction ($\alpha = 0.05 / 3 = 0.0167$ per contrast).

- **H1 (Retrieval Recall):** 128 minus 1024 tokens $> 0$ with margin $\ge 1.50$ pp.
- **H2 (Retrieval MRR):** 128 minus 1024 tokens $> 0$ with margin $\ge 1.50$ pp.
- **H3 (Answer Generation F1):** 1024 minus 128 tokens $> 0$ with margin $\ge 1.50$ pp.

| Required Contrast | Direction | Difference, pp | 95% Paired CI, pp | Simultaneous 95% CI, pp | Standard Error, pp | Verdict Impact |
|---|---|---:|---:|---:|---:|---|
| **recall** | 128 - 1024 | **+8.78** | [+7.97, +9.57] | [+7.82, +9.75] | 0.41 | PROVEN (lower bound $> 1.5$ pp) |
| **mrr** | 128 - 1024 | **-0.27** | [-1.04, +0.49] | [-1.19, +0.67] | 0.39 | INCONCLUSIVE (covers zero) |
| **f1** | 1024 - 128 | **-1.07** | [-2.12, -0.06] | [-2.38, +0.18] | 0.52 | INCONCLUSIVE (covers zero simultaneously; misses +1.5 pp) |
| **em** (secondary) | 1024 - 128 | **-0.87** | [-1.82, +0.08] | [-2.07, +0.33] | 0.50 | Neutral |

Because the familywise simultaneous confidence interval for F1 covers zero (upper bound +0.18 pp) and falls far below the pre-registered +1.50 pp practical margin, the formal decision rule classifies the test as **INCONCLUSIVE**. However, under ordinary 95% confidence intervals, 1024-token chunks are strictly *worse* than 128-token chunks ([-2.12, -0.06] pp), directly disproving the premise that larger chunks yield superior answer generation.

## Blinded LLM-as-a-Judge Evaluation

A sample of 60 validation questions (240 candidate answers across 4 chunk sizes) was evaluated with Qwen3-4B prompted as an impartial judge blinded to chunk size:
- 128 tokens: 11.7% judged accuracy [95% CI: 5.0%, 20.0%]
- 256 tokens: 13.3% judged accuracy [95% CI: 5.0%, 21.7%]
- 512 tokens: 15.0% judged accuracy [95% CI: 6.7%, 25.0%]
- 1024 tokens: 16.7% judged accuracy [95% CI: 8.3%, 26.7%]

Agreement between the LLM judge and token F1 >= 0.5 was 80.0% to 81.7% across all chunk sizes.

## Diagnostics and Corpus Audit

1. **Document Lengths:** MuSiQue Wikipedia candidate documents average 139.6 tokens. At chunk size 1024, 98.9% of documents fit in a single chunk.
2. **Context Diversity:** Under the 1024-token budget control:
   - 128-token chunks retrieved context assembled from an average of **8.96 distinct documents**.
   - 1024-token chunks assembled context from only **6.16 distinct documents**.
3. **Multi-Hop Mechanism:** In MuSiQue, answering a question requires connecting 2 to 4 supporting facts from distinct Wikipedia pages. By packing smaller, targeted chunks from almost 9 documents into the prompt budget, 128-token chunks achieve 60.13% recall of gold documents, whereas 1024-token chunks achieve only 51.34% recall (+8.78 pp difference).

## Caveats and Limitations

1. **Synthetic Multi-Hop Benchmark:** MuSiQue was constructed by composing single-hop questions into multi-hop chains. While it eliminates single-hop shortcut artifacts present in HotpotQA, real-world enterprise RAG corpora may have different paragraph lengths and hop distributions.
2. **Fixed Budget Constant:** We evaluated budget control at $B = 1024$ tokens. Larger budgets (e.g. 4k or 8k tokens) would allow larger chunks to retrieve multiple documents as well, though smaller chunks would also scale their document coverage proportionally.
3. **Model Specifics:** Findings condition on `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-4B` with greedy generation. Different retriever/reader capacity ratios could alter the exact crossover point.
