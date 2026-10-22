# MuSiQue: Does RAG Chunk-Size Ranking Flip on Multi-Hop QA (MuSiQue)?

## Question and Evidence
In retrieval-augmented generation (RAG), does smaller chunk size improve retrieval precision (Recall and MRR) while larger chunk size improves answer generation (Token F1), creating a metric ranking flip?

- **Source claim:** [Aqeel Abbas, "Which RAG Chunk Size Is Best? The Answer Flips Depending on How You Score It"](https://medium.com/illumination/which-rag-chunk-size-is-best-the-answer-flips-depending-on-how-you-score-it-2c3ef424e50c). Abbas argued that 128-token chunks win on retrieval precision while 1024-token chunks win on answer quality.
- **[the original HotpotQA run](../original-run/) findings (HotpotQA):** Evaluated 4 chunk sizes (128, 256, 512, 1024) on 600 HotpotQA questions with fixed top-k=5. Found +2.10 pp F1 for 1024 vs 128 (simultaneous CI [-0.34, +4.62] pp), but retrieval differences were negligible (-0.08 pp Recall, -0.03 pp MRR). Crucially, HotpotQA paragraphs averaged only ~115 tokens, meaning 256, 512, and 1024 chunks were 98% identical, and fixed top-k gave 1024 an unfair 8x token budget advantage.
- **[the budget-controlled multi-hop study](../chunk-size-multihop-fresh/) findings (Budget-Controlled Multi-Hop):** Controlled the reading token budget across 4,200 questions. Found +1.49 pp F1 (1024 vs 128 tokens, budget-controlled), 95% CI [+0.19, +2.76] pp, falling just 0.01 pp short of the 1.5 pp pre-registered bar: verdict was **INCONCLUSIVE**.
- **MuSiQue study goal:** Test on a fresh, third multi-hop benchmark not previously evaluated: **MuSiQue** (`dgslibisey/MuSiQue`, answerable split). MuSiQue was specifically designed to prevent single-hop shortcuts and test complex 2-4 hop reasoning over 20 candidate passages.

## Power Calculation and Pre-Registration
From [the budget-controlled multi-hop study](../chunk-size-multihop-fresh/):
- Sample size: $N = 4,200$
- Mean paired difference ($F1_{1024} - F1_{128}$): $+1.49$ percentage points
- 95% confidence interval: $[+0.19, +2.76]$ pp
- Confidence interval half-width: $HW = (2.76 - 0.19) / 2 = 1.285$ pp
- Standard error: $SE = 1.285 / 1.96 = 0.6556$ pp
- Paired standard deviation: $s_d = SE \times \sqrt{4200} = 0.6556 \times 64.8074 \approx 42.49$ pp (consistent with [the original HotpotQA run](../original-run/)'s pilot $s_d \approx 43.6$ pp).

To detect a practical margin of $\delta = 1.50$ pp at two-sided $\alpha = 0.05$ with 80% power ($z_{\alpha/2} = 1.960, z_\beta = 0.842$):
$$N = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot s_d^2}{\delta^2} = \frac{(1.960 + 0.842)^2 \cdot (42.49)^2}{(1.5)^2} = \frac{7.851 \times 1805.4}{2.25} \approx 6,299 \text{ questions}$$

MuSiQue's full validation benchmark contains exactly **2,417 answerable questions**.
At $N = 2,417$:
- Expected standard error: $SE = 42.49 / \sqrt{2417} \approx 0.864$ pp
- Expected 95% CI half-width: $1.96 \times 0.864 \approx 1.69$ pp
- Statistical power to detect 1.50 pp at $\alpha = 0.05$: $z = (1.50 - 1.69) / 0.864 \approx -0.22$, which yields $\Phi(-0.22) \approx 41\%$.

**Pre-registration designation:** Because $N = 2,417$ is bounded by the official MuSiQue validation split and the 8 Colab CU budget, this experiment is explicitly pre-registered as an **estimation study**. It is sized to provide a high-precision estimation (95% CI half-width $\approx \pm 1.7$ pp) of the true chunk-size effect on MuSiQue and determine whether the +1.49 pp effect observed in [the budget-controlled multi-hop study](../chunk-size-multihop-fresh/) replicates or attenuates.

## Pre-Registered Hypotheses and Contrasts
1. **H1 (Retrieval Recall):** 128-token chunks beat 1024-token chunks on Document Recall (fraction of gold supporting documents present in retrieved context). Direction: $Recall_{128} - Recall_{1024} > 0$.
2. **H2 (Retrieval MRR):** 128-token chunks beat 1024-token chunks on Mean Reciprocal Rank (MRR). Direction: $MRR_{128} - MRR_{1024} > 0$.
3. **H3 (Answer Generation F1):** 1024-token chunks beat 128-token chunks on token F1 under budget-controlled reading context. Direction: $F1_{1024} - F1_{128} > 0$.

### Practical Margin
A fixed practical threshold of **1.5 percentage points** (0.015 absolute) is required for each primary metric, matching [the budget-controlled multi-hop study](../chunk-size-multihop-fresh/).

### Fixed Decision Rule (Identical to [the budget-controlled multi-hop study](../chunk-size-multihop-fresh/))
Using 10,000 paired question bootstrap resamples:
- **PROVEN:** If all three primary simultaneous lower confidence bounds (Bonferroni $\alpha = 0.05 / 3 = 0.0167$, 98.33% coverage per contrast) are $> 0$ AND every point difference is $\ge 1.50$ percentage points.
- **DISPROVEN:** If any primary simultaneous upper confidence bound is $< 0$ (a required direction significantly reverses).
- **INCONCLUSIVE:** Otherwise (confidence interval covers zero, or point estimate falls short of the 1.5 pp practical threshold).

## Experimental Setup
- **Dataset:** MuSiQue (`dgslibisey/MuSiQue`, answerable split).
  - Pilot: 300 questions randomly sampled from MuSiQue `train` (seed 132).
  - Confirmation: all 2,417 questions from MuSiQue `validation` (seed 132). Fresh items, no overlap.
- **Corpus:** Deduplicated pool of Wikipedia context documents associated with the questions.
- **Chunk Sizes:** 128, 256, 512, and 1024 tokens.
- **Embedding Model:** `Qwen/Qwen3-Embedding-0.6B`, last-token pooling, normalized L2 vectors, exact FAISS inner-product retrieval.
- **Reader Model:** `Qwen/Qwen3-4B`, greedy decoding (temperature 0), max 64 new tokens, thinking disabled.
- **Budget Control:** The context token budget provided to the LLM reader is controlled at $B = 1024$ tokens total context across all chunk sizes.
  - 128-token chunks: top chunks retrieved up to 1024 context tokens.
  - 256-token chunks: top chunks retrieved up to 1024 context tokens.
  - 512-token chunks: top chunks retrieved up to 1024 context tokens.
  - 1024-token chunks: top chunks retrieved up to 1024 context tokens.
  Additionally, fixed top-k (k=5) is evaluated as a secondary comparison to measure the confounding impact of unbudgeted context.
- **Evaluation Metrics:**
  - Token F1 (max over gold answer aliases, normalized).
  - Exact Match (EM).
  - Document Recall and MRR over gold supporting facts.
  - Total prompt and completion tokens recorded from the model.

## Repository contents
- `README.md`
- `writeup.md`
- `blog.md`
- `results/` (raw JSONL, summary.json, audit.json, timing.json)
- `figures/` (charts, table.png, diagram.png, colab_run.txt)
- `figures/screenshots/` (evidence panels rendered from saved files via `scripts/render_evidence.py`)

## Decisions
- **Dataset Selection:** Selected MuSiQue (`dgslibisey/MuSiQue`, answerable validation split, $N = 2,417$ questions) over 2WikiMultihopQA because MuSiQue specifically removes single-hop reasoning shortcuts by design and enforces connected reasoning across multiple hops.
- **Budget Control ($B = 1024$ tokens):** In [the original HotpotQA run](../original-run/) (HotpotQA), fixed top-k gave 1024-token chunks an 8x token advantage (~5,000 prompt tokens vs ~730 prompt tokens). In this MuSiQue study, we controlled total context tokens at 1024 across all chunk sizes to isolate chunking granularity from total reading budget.
- **Resumption Strategy:** Evaluated pilot (300 items) and confirmatory items incrementally on Colab L4. Generation of all 2,417 questions across all 4 chunk sizes (9,668 answers) and 240 blinded judge evaluations were completed cleanly with full checkpointing.
- **Diagram Generation:** The pipeline visualization was created with matplotlib at 150 dpi in `figures/diagram.png`.

## Confirmation Results Summary
- **Verdict:** **INCONCLUSIVE** (pre-registered decision rule required all 3 simultaneous lower bounds $>0$ and point difference $\ge 1.50$ pp).
- **Headline Number:** **-1.07 percentage points** answer token F1 for 1024 minus 128 tokens (95% CI [-2.12, -0.06] pp; simultaneous 95% CI [-2.38, +0.18] pp).
- **Key Metrics (Confirmation N = 2,417):**
  - Chunk 128: Recall 60.13% [59.00, 61.26], MRR 82.23%, F1 13.88% [12.70, 15.12], EM 8.15%
  - Chunk 256: Recall 56.44% [55.32, 57.57], MRR 82.85%, F1 13.56% [12.37, 14.77], EM 7.90%
  - Chunk 512: Recall 53.49% [52.38, 54.62], MRR 83.12%, F1 12.63% [11.47, 13.82], EM 7.36%
  - Chunk 1024: Recall 51.34% [50.24, 52.47], MRR 82.50%, F1 12.81% [11.66, 14.02], EM 7.28%
- **Contrasts:**
  - Recall (128 - 1024): **+8.78 pp** [95% CI: +7.97, +9.57] [Simultaneous CI: +7.82, +9.75], SE=0.41 pp
  - MRR (128 - 1024): **-0.27 pp** [95% CI: -1.04, +0.49] [Simultaneous CI: -1.19, +0.67], SE=0.39 pp
  - F1 (1024 - 128): **-1.07 pp** [95% CI: -2.12, -0.06] [Simultaneous CI: -2.38, +0.18], SE=0.52 pp
  - EM (1024 - 128): **-0.87 pp** [95% CI: -1.82, +0.08] [Simultaneous CI: -2.07, +0.33], SE=0.50 pp

## Saved Evidence Panels
1. `figures/screenshots/01_first_run_hotpotqa.png`: The original HotpotQA report from [the original HotpotQA run](../original-run/) showing +2.10 pp nominal F1 gain for 1024-token chunks under unbudgeted top-5 retrieval.
2. `figures/screenshots/02_evidence_of_flaw_audit.png`: Saved audit of chunk length distribution showing 99.7% of HotpotQA documents fit into a single chunk, producing 98.0% identical retrieved documents.
3. `figures/screenshots/03_fixed_run_output.png`: Saved execution log showing Qwen3-4B generation across all 2,417 items for 1024 chunks and final blinded judge completion on Colab L4.
4. `figures/screenshots/04_final_stats_verdict.png`: Saved summary showing final confirmation statistics, simultaneous bootstrap intervals, and the INCONCLUSIVE verdict.
5. `figures/screenshots/05_colab_l4_gpu_run.png`: Saved `nvidia-smi` output and timestamps on Colab L4 confirming execution environment and GPU specs.

## Timing
- Timing: estimated 120 minutes; actual 262 minutes elapsed (setup: 20 min, GPU run: 205 min, eval: 15 min, writeup: 22 min); NVIDIA L4; 4.1 Colab CU used (6.05 CU shared-account upper bound).

To regenerate figures from saved statistics without recomputing results, run `python scripts/analyze.py --figures-only`.
