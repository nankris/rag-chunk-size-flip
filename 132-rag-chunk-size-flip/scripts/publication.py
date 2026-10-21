"""Render the final narrative from audited results, without changing hypotheses."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'results'
s=json.loads((O/'summary.json').read_text());a=json.loads((O/'audit.json').read_text());t=json.loads((O/'timing.json').read_text());u=json.loads((O/'usage_final.json').read_text())
p=s['pilot'];c=s['confirm'];v=c['verdict']
def val(st,sz,m):return s[st]['metrics'][str(sz)][m]['mean']
def pct(x):return f'{100*x:.1f}'
def ci(z,key='ci95'):return f"[{100*z[key][0]:.1f}, {100*z[key][1]:.1f}]"
def ci2(z,key='ci95'):return f"[{100*z[key][0]:.2f}, {100*z[key][1]:.2f}]"
def cell(st,sz,m):
 z=s[st]['metrics'][str(sz)][m];return f"{pct(z['mean'])} {ci(z)}"
headline=f"{pct(c['contrasts']['f1']['mean'])} percentage points"
timing=f"Timing: estimated 100 minutes; actual {t['actual_minutes']:.1f} minutes elapsed, including recovery/waiting; NVIDIA L4; at most {u['conservative_account_decrease_CU']:.2f} CU (shared-account upper bound)."
table='| Dataset | Chunk tokens | Recall@5 % [95% CI] | MRR@5 % [95% CI] | Answer F1 % [95% CI] | Exact match % [95% CI] | Mean prompt tokens |\n|---|---:|---:|---:|---:|---:|---:|\n'
for st in ['pilot','confirm']:
 for sz in [128,256,512,1024]:
  table+='| '+('SQuAD pilot' if st=='pilot' else 'HotpotQA confirm')+f' | {sz} | '+' | '.join(cell(st,sz,m) for m in ['recall','mrr','f1','em'])+f" | {val(st,sz,'prompt_tokens'):,.0f} |\n"
contr='| Required contrast | Difference, points | 95% paired CI | Familywise 95% CI |\n|---|---:|---:|---:|\n'
for m in ['recall','mrr','f1']:
 z=c['contrasts'][m];contr+=f"| {m}: {z['direction']} | {100*z['mean']:.2f} | {ci2(z)} | {ci2(z,'ci_simultaneous')} |\n"
if v=='DISPROVEN':take='The preregistered claim that small chunks improve retrieval while large chunks improve answers failed on the fresh confirmation dataset. A chunk-size winner depends on this corpus and operating policy; this run does not establish a universal ranking flip.'
elif v=='PROVEN':take='Small chunks improved retrieval while large chunks improved answer F1 on the fresh confirmation dataset, meeting the fixed test for a reversal. This result applies to the tested top-five policy and corpus, where chunk size also changes the amount of text the model reads.'
else:take='This run did not resolve the preregistered ranking-flip claim at the chosen confidence and practical thresholds. The observed differences are useful measurements, but they do not establish a reliable universal reversal.'
source='https://medium.com/illumination/which-rag-chunk-size-is-best-the-answer-flips-depending-on-how-you-score-it-2c3ef424e50c'
winners={}
for metric in ['recall','mrr','f1','em']:
 best=max(val('confirm',sz,metric) for sz in [128,256,512,1024]);winners[metric]=', '.join(str(sz) for sz in [128,256,512,1024] if abs(val('confirm',sz,metric)-best)<1e-12)
judge_table='| Chunk tokens | Qwen judged accuracy % [95% CI], n=60 |\n|---|---:|\n'
for sz in ['128','256','512','1024']:
 z=c['judge'][sz];judge_table+=f"| {sz} | {100*z['accuracy']:.1f} {ci(z)} |\n"
readout=f"""# Experiment 132: Does the RAG chunk-size winner flip?

**Verdict: {v}.** Headline number: **{headline}** answer token F1 for 1024 minus 128 tokens on 600 fresh HotpotQA questions.

{take}

{timing}

## Results

{table}

Descriptive four-size point-estimate leaders on confirmation: Recall@5 {winners['recall']}; MRR@5 {winners['mrr']}; F1 {winners['f1']}; exact match {winners['em']} tokens. These are not additional significance tests.

Scores are macro averages over questions. All score intervals use 10,000 paired question bootstrap draws, seed 132. The table is also rendered in `figures/table.png` for Substack.

## Fixed decision rule and confirmation

H1 predicted 128 > 1024 on both document Recall@5 and MRR@5. H2 predicted 1024 > 128 on token F1. PROVEN requires every simultaneous lower bound >0 and every point gain >=5 points; DISPROVEN requires any simultaneous upper bound <0; otherwise INCONCLUSIVE. The five-point threshold represents one more useful result per 20 questions. Exact match and judging are secondary.

{contr}

The pilot alone would be labelled DISPROVEN by the same directional rule: large-minus-small F1 was {100*p['contrasts']['f1']['mean']:.2f} points, simultaneous CI {ci2(p['contrasts']['f1'],'ci_simultaneous')}. It did not establish the predicted answer benefit. The final verdict uses confirmation only.

On confirmation, the ordinary F1 interval barely excludes zero, but the simultaneous interval crosses it and the 2.10-point gain misses the fixed five-point practical threshold. The planned retrieval advantages are absent. This is why a nominally positive answer result does not prove the full claim.

Pilot paired standard deviations were {p['contrasts']['recall']['paired_sd']:.3f} for Recall, {p['contrasts']['mrr']['paired_sd']:.3f} for MRR, and {p['contrasts']['f1']['paired_sd']:.3f} for F1. At the already-fixed confirmation n=600, these imply approximate simultaneous half-widths of {100*p['contrasts']['recall']['planned_confirmation_simultaneous_halfwidth']:.1f}, {100*p['contrasts']['mrr']['planned_confirmation_simultaneous_halfwidth']:.1f}, and {100*p['contrasts']['f1']['planned_confirmation_simultaneous_halfwidth']:.1f} percentage points if paired variability transferred across datasets. They are precision diagnostics, not guaranteed power on another dataset.

Bonferroni 98.333% individual intervals provide at least 95% simultaneous coverage over the three planned directional contrasts. No hypothesis or sample size changed after pilot outcomes. Preregistration commit: `554321b`.

## Method and diagnostics

Pilot: 300 SQuAD questions, 48 title-grouped articles. Confirmation: 600 HotpotQA distractor validation questions and {a['confirm']['documents']:,} title-deduplicated context documents, searched globally. Seed 132 fixes selection. Chunking uses nonoverlapping Qwen-tokenizer windows; embeddings use Qwen3-Embedding-0.6B with normalized last-token pooling and full chunk text. FAISS performs exact inner-product search. Qwen3-4B answers greedily from top five chunks, thinking off, at most 64 output tokens. Actual served-model input/output counts are in every prediction record. Model revisions and package versions are saved.

The confirmation corpus has one duplicate-title text conflict, resolved by taking the first version before retrieval. Document relevance is a coarse label: a chunk from a gold document can score as relevant without containing the answer. HotpotQA Recall counts both supporting documents; MRR rewards the first one. These definitions can disagree with useful evidence retrieval.

At 512 versus 1024 tokens, identical top-five document lists occur on {100*a['confirm']['identical_512_1024_top5_document_lists_fraction']:.1f}% of confirmation questions. At 512 tokens, {100*a['confirm']['sizes']['512']['single_chunk_documents_fraction']:.1f}% of documents fit a single chunk. Mean chunk lengths are {a['confirm']['sizes']['512']['mean_chunk_tokens']:.1f} and {a['confirm']['sizes']['1024']['mean_chunk_tokens']:.1f} tokens respectively. A configured maximum is not a guarantee that documents fill it.

Qwen judged 60 questions per dataset at all four sizes, 240 candidate answers per dataset. Confirmation valid labels: {c['judge_total']['valid']}/240. Agreement with F1 >=0.5: {100*c['judge_total']['agreement_f1_half']:.1f}%; agreement with exact match: {100*c['judge_total']['agreement_em']:.1f}%. This uses the same model family and reference answers, not human adjudication. Disagreements are saved in `results/judge_disagreements.json`.

{judge_table}

## Caveats and reproducibility

This is an independent test inspired by [Aqeel Abbas's article]({source}); only its preview was accessible. We cannot claim to reproduce its hidden methods. Fixed top-five retrieval changes the reading budget across sizes, and HotpotQA snippets are often shorter than the largest chunk limits. This is not a full Wikipedia retrieval benchmark. The intervals condition on one corpus and model, ignore document clustering, and do not quantify model-seed or training uncertainty. SQuAD has only 48 articles, making its question-level intervals particularly optimistic for cross-document generalization. Public benchmarks may have appeared in pretraining. F1 can penalize valid paraphrases; the sampled model judge does not eliminate that issue.

All 3,600 answers and 480 judge calls were checkpointed. `scripts/validate.py` independently recomputes retrieval labels and checks completeness; `scripts/analyze.py`, `scripts/export.py`, and `scripts/publication.py` reproduce statistics, charts, tables, and this report. Resume via `scripts/sync.sh` while the session exists. Raw data, prompts' ingredients, predictions, scores, token counts, configs, and runtime logs are retained. See `results/usage_final.json` for honest shared-account CU accounting and `results/timing.json` for timing limitations.
"""
(R/'writeup.md').write_text(readout)
# A standalone post, with no hidden dependence on this conversation.
blog=f"""# I tested whether the best RAG chunk size changes when you score the answers

A search system can look good before anyone reads its answers. That gap is the reason I wanted to test a deceptively small RAG setting: chunk size.

My headline result is **{headline}**. That is the change in answer token F1 when I moved from 128-token chunks to 1024-token chunks on a fresh set of 600 HotpotQA questions. The verdict under my fixed decision rule is **{v}**.

{take}

The idea came from [Aqeel Abbas's article, “Which RAG Chunk Size Is Best? The Answer Flips Depending on How You Score It.”]({source}) Its accessible preview describes the tension between precise passages and surrounding context. The detailed experiment is behind a paywall. This post builds on that question, without claiming to reproduce a method I could not inspect.

I wanted to make the prediction specific enough to fail. Before running the tests, I wrote down two claims. First, 128-token chunks would beat 1024-token chunks on retrieval Recall@5 and reciprocal rank. Second, 1024-token chunks would beat 128-token chunks on answer F1. Together, those claims would produce the proposed ranking reversal.

I also tested 256 and 512 tokens. Those intermediate settings help explain the shape of the result, but I did not let them replace the planned endpoint comparison after seeing the scores.

## What went into the test

The pilot used 300 questions from SQuAD. I combined its validation paragraphs into 48 article documents, then searched those documents for every question. To reduce repeated-context dependence, the sample contained one question per source paragraph.

The confirmation used a different dataset: 600 HotpotQA validation questions. I combined the context documents supplied with those examples into a corpus of {a['confirm']['documents']:,} documents. Every question searched that whole corpus. The retriever did not get a private list of ten documents for its own question.

This is still a compact benchmark built from supplied snippets. It is not retrieval over all of Wikipedia. That distinction matters when interpreting both the scores and the meaning of a 1024-token chunk.

For each dataset, I kept documents and questions fixed across all four chunk sizes. Chunks were consecutive windows without overlap. Qwen3-Embedding-0.6B embedded the full text, and FAISS searched normalized vectors using exact inner products.

Qwen3-4B then answered from the five highest-ranked chunks. It used greedy decoding with thinking disabled and a short-answer instruction. Each answer had a maximum of 64 generated tokens. The saved records include actual input and output token counts from the model.

## What the scores say

![Main results with bootstrap intervals](figures/table.png)

The pilot's retrieval scores favored small chunks. Recall@5 was {pct(val('pilot',128,'recall'))}% at 128 tokens and {pct(val('pilot',1024,'recall'))}% at 1024. Pilot answer F1 was {pct(val('pilot',128,'f1'))}% and {pct(val('pilot',1024,'f1'))}% respectively.

On the fresh HotpotQA sample, Recall@5 was {pct(val('confirm',128,'recall'))}% for 128 tokens and {pct(val('confirm',1024,'recall'))}% for 1024. Reciprocal rank was {pct(val('confirm',128,'mrr'))}% versus {pct(val('confirm',1024,'mrr'))}%. Answer F1 was {pct(val('confirm',128,'f1'))}% versus {pct(val('confirm',1024,'f1'))}%, with exact match at {pct(val('confirm',128,'em'))}% versus {pct(val('confirm',1024,'em'))}%.

Across all four settings, the point-estimate leaders were {winners['recall']} tokens for recall, {winners['mrr']} for reciprocal rank, and {winners['f1']} for answer F1. These descriptive winners do not override the preregistered comparison.

Those are different questions about the same pipeline. Recall measures how much of the relevant document set appears among the five retrieved chunks. Reciprocal rank rewards finding a relevant document earlier. F1 compares words in the generated answer with words in the reference answer.

A crucial limitation is that my retrieval labels identify source documents. A chunk can come from the right document while omitting the sentence needed to answer. That means a strong retrieval score is not a certificate that the generator received the necessary evidence.

![All four metrics across chunk sizes](figures/metrics.png)

## How I decided what counted

I used paired bootstrap intervals, resampling questions 10,000 times. Each resample preserved the comparison between chunk sizes on the same questions.

For the three primary claims, I adjusted the intervals so their combined coverage was at least 95%. I also required a gain of at least five percentage points on every planned metric before calling the full claim proven. That threshold corresponds to one more useful result per twenty questions, a practical reason to consider changing ingestion settings.

The confirmation F1 difference had a conventional 95% interval of {ci(c['contrasts']['f1'])} percentage points. Its simultaneous interval was {ci(c['contrasts']['f1'],'ci_simultaneous')}. The ordinary interval barely clears zero, but the adjusted interval does not. The gain also falls below the fixed five-point threshold, while the predicted retrieval advantage is absent. The verdict follows the complete preregistered rule.

![The planned directional comparisons](figures/contrasts.png)

## The reading budget complicates the story

Five chunks is a fixed count, but it is not a fixed amount of text. Mean confirmation prompt length rose from {val('confirm',128,'prompt_tokens'):,.0f} tokens at the smallest setting to {val('confirm',1024,'prompt_tokens'):,.0f} at the largest. Chunk size therefore changes the generator's reading budget as well as the retrieval units.

There is another wrinkle. Many HotpotQA documents are short snippets. Increasing the maximum from 512 to 1024 does not make those documents longer. The two settings returned identical top-five document lists on {100*a['confirm']['identical_512_1024_top5_document_lists_fraction']:.1f}% of questions. The configured chunk limit and the text actually retrieved deserve separate attention.

![Prompt budget versus answer F1](figures/tokens_vs_f1.png)

## A check on answer scoring

Word overlap can punish an acceptable paraphrase or reward an incomplete answer. I therefore asked Qwen to judge 60 questions per dataset at all four sizes against reference answers.

On confirmation, the judge agreed with the F1-at-least-0.5 classification {100*c['judge_total']['agreement_f1_half']:.1f}% of the time. It agreed with exact match {100*c['judge_total']['agreement_em']:.1f}% of the time. I saved disagreements for inspection. A pilot disagreement illustrates the gap: “5” versus “five” received zero lexical F1 but was accepted by the judge. Another judge call accepted “about 11,000 years ago” against “11,700 years ago,” showing that model grading can be permissive too. This is a diagnostic from the same model family, not independent human validation.

The remaining limits are substantial: one embedding model, one answer model, one prompt, no overlap, and one fixed top-five policy. The intervals treat questions as the sampling units, even though questions can share documents. They do not measure uncertainty across model training runs. Public benchmark familiarity is also possible.

For engineers, the useful result is a measured comparison with an explicit failure condition. Check both evidence retrieval and final answers on your workload, record the actual tokens read, and inspect the document lengths before interpreting a chunk-size sweep. This experiment supplies evidence for one operating policy, with its misses left visible.

{timing}
"""
(R/'blog.md').write_text(blog)
print('blog words',len(blog.split()),'verdict',v)
