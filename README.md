# rag-chunk-size-flip

Experiments 132, 132b and 132c (RAG chunk size on HotpotQA, fresh multi-hop questions and MuSiQue). Experiment arms in this repo. Each folder has its own README, code, raw results, chart and verdict.

- [`132-rag-chunk-size-flip`](132-rag-chunk-size-flip/) Experiment 132: Does RAG chunk-size ranking flip with the metric? (128 vs 1024 tokens, 600 HotpotQA questions) — **INCONCLUSIVE**
- [`132b-chunk-size-multihop-fresh`](132b-chunk-size-multihop-fresh/) Experiment 132b: Bigger chunks on fresh multi-hop questions (budget-controlled) — **INCONCLUSIVE**
- [`132c-chunk-size-musique`](132c-chunk-size-musique/) Experiment 132c: Does the chunk-size ranking flip on MuSiQue multi-hop QA? — **INCONCLUSIVE**

Large artifacts excluded from git are attached to release [`v1-data`](https://github.com/nankris/rag-chunk-size-flip/releases/tag/v1-data); see [RELEASE_ASSETS.md](RELEASE_ASSETS.md).
