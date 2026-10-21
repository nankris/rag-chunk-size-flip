# Experiment 132b: bigger chunks on fresh multi-hop questions

## Question and preregistered prediction
Does increasing the chunk limit from 128 to 1024 tokens improve multi-hop answer F1 by at least 1.5 percentage points when evidence tokens are controlled?

H1: 1024 minus 128 answer F1 has a simultaneous confidence interval lower bound above zero and a point estimate >=1.5 points under the fixed evidence budget. This is the requested operational rule; it does not establish that the population gain itself exceeds 1.5 points. The experiment title's "2 points" is a rounded motivation, not a substituted decision threshold. The 1.5-point margin means about 15 extra fully correct answer equivalents per 1,000 requests and is the user-specified follow-up margin.

Evidence: #132 found +2.10268 points on 600 HotpotQA questions, simultaneous CI [-0.34,4.62], missing its five-point threshold. This historical pilot motivates the direction and supplies the paired per-question SD, 0.2583962030. It is not reused for confirmation. Source inspiration: [Aqeel Abbas, Which RAG Chunk Size Is Best?](https://medium.com/illumination/which-rag-chunk-size-is-best-the-answer-flips-depending-on-how-you-score-it-2c3ef424e50c). Only the public preview is accessible, so this is an independent follow-up.

## Fixed setup, sample size, and analysis
- Confirmation: public HF `cmriat/2wikimultihopqa`, validation, seed 1320132, 3,600 shuffled questions; reserve the first 16 shuffled questions for pipeline smoke checks and exclude them from confirmation. No confirmatory answer results are inspected to tune this protocol.
- Corpus: all context paragraphs of selected confirmation questions, globally searched, deduplicated by exact title AND text. Distinct versions remain distinct documents. Supporting facts matched to each question's original paragraph version and sentence token spans. No synthetic padding or merging unrelated paragraphs to manufacture long documents.
- Same family as #132: nonoverlapping Qwen-tokenizer windows, 128/256/512/1024 maxima; Qwen3-Embedding-0.6B last-token normalized embeddings, FAISS exact inner product. Qwen3-8B tokenizer used for chunk and reader accounting. Record actual chunk lengths, one-chunk fractions and duplicate retrievals.
- Main regime: 1,024 evidence tokens, greedily consume ranked chunks until budget reached, truncate only the last chunk. This gives variable k, recorded per question. Rank up to 128 chunks; if insufficient text, report shortfall. Evidence text excludes passage-label/prompt overhead, measured separately with actual reader token counts. Secondary regime: equal k=5 untruncated chunks. Same ranking, corpus, question and prompt in both regimes.
- Reader: Qwen/Qwen3-8B on vLLM, L4, FP16, greedy temperature 0, thinking disabled, max 64 output tokens, fixed seed 1320132. Actual served-model input and output token counts per answer. Prompt requests only a short answer from passages, or unknown.
- Primary metric: macro answer token F1 using lowercase, article/punctuation/whitespace normalization and special yes/no handling, maximum over provided aliases. Secondary: EM, supporting-fact recall (a sentence must be fully contained in the union of selected token spans), support-document recall, evidence and reader tokens.
- Six planned contrasts: each of 256,512,1024 minus 128 F1 in each of the two regimes. 20,000 paired question bootstrap draws, seed 1320132. Bonferroni two-sided 99.1667% percentile intervals give >=95% simultaneous coverage over these six contrasts. Report ordinary 95% marginal intervals on arm scores and secondary metrics, clearly distinguished.
- n=ceil(((z_(1-.05/12)+z_.8)*0.2583962030/.015)^2)=3,594; round up to 3,600. This approximates 80% power for excluding zero at a true 1.5-point effect, assuming historical variance transfers. It does NOT promise 80% probability of also clearing the point-estimate margin at its boundary. Question bootstrap conditions on this corpus and one reader seed; document/topic dependence is a limitation.
- Verdict: PROVEN if H1's simultaneous lower bound >0 AND observed gain >=1.5 pp. DISPROVEN if its simultaneous upper bound <1.5 pp (evidence against the practical margin). Otherwise INCONCLUSIVE. Equal-k and intermediate-size results cannot rescue H1. No post-result changes to hypothesis, sample size, margin, or endpoint.
- If the CU cap prevents completion, report the incomplete run as INCONCLUSIVE, never as the full preregistered test. Checkpoints are resumable; no repeated completed inference.

## Execution and deliverables
Run only in this directory and named Colab session `b9-132b`. GPU jobs run detached on the VM with incremental JSONL files, logs and copies back. CU cap 6; account usage differences are conservative shared-account bounds. Always stop the session at completion or failure. HF data only, no paid inference APIs.

Timing: estimated 110 minutes (setup 20, GPU run 65, analysis 10, writing 15); actual 105.0 minutes. GPU: NVIDIA L4; CU ≤4.93 (conservative accounting bound). Timing starts in `results/timing.json` before setup work. Local git commits retain preregistration, scripts, data, results, figures and final reports. See writeup.md and blog.md after completion.

## Reproduction commands (implementation notes, not a protocol revision)

The saved package snapshot includes vLLM 0.32.0, PyTorch 2.13.0 and Transformers 5.18.0 on the allocated Colab image. The initial install was `pip install vllm faiss-cpu`; exact resolved versions are in `results/packages.txt`. Upload this directory's scripts and frozen configuration to `/content/132b`, then run `bash scripts/run.sh` as a detached process with stdout/stderr directed to `run.log`. `scripts/launch.py` shows the detached launcher, including duplicate-process checks. No GPU work runs in the sandbox.

Use `scripts/RESUME.md` before any restart. To reproduce the saved analysis locally, run `/workspace/tools/mlenv/bin/python scripts/validate.py`, then `scripts/analyze.py` and `scripts/figures.py` using that Python. Frozen question text, gold answers and context versions are retained in `results/data.json`. This implementation section was added while embedding was running, before reader outcomes existed; the registered setup and statistical decision above are unchanged.

## Completed result

**INCONCLUSIVE**: budget-controlled 1024 minus 128 F1 = **+1.49 pp**, simultaneous CI **[+0.19, +2.76] pp**. The original prediction and decision rule above are preserved.

All 3,600 questions and 28,800 arm records passed validation. Qwen3-8B served 20,740 unique prompts; exact token totals are in results/summary.json. Named Colab session b9-132b is stopped.

See [writeup.md](writeup.md) for numerical results and caveats, [blog.md](blog.md) for the publication draft, and figures/ for PNGs, editable diagram and terminal evidence. Local scripts reproduce the analysis without a GPU.
