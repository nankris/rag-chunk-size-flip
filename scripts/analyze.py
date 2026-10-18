"""Fixed paired bootstrap, audits and publication charts. Run locally after sync."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';FIG=ROOT/'figures'
SIZES=[128,256,512,1024];METRICS=['recall','mrr','f1','em'];LABELS={'recall':'Recall@5','mrr':'MRR@5','f1':'Answer token F1','em':'Exact match'}
def records(p):return [json.loads(l) for l in p.read_text().splitlines()]
def interval(x,ix,alpha=.05):
 b=x[ix].mean(axis=1);return [float(np.quantile(b,alpha/2)),float(np.quantile(b,1-alpha/2))]
def main():
 summary={};audit={};allrows={}
 for stage in ['pilot','confirm']:
  data=json.loads((OUT/f'{stage}_data.json').read_text());ids=[q['id'] for q in data['questions']];n=len(ids)
  rng=np.random.default_rng(132);ix=rng.integers(n,size=(10000,n));rows={}
  for size in SIZES:
   rr=records(OUT/f'{stage}_answers_{size}.jsonl');rmap={r['id']:r for r in rr};assert len(rmap)==n==len(rr)
   rows[size]=[rmap[i] for i in ids]
  allrows[stage]=rows;summary[stage]={'n':n,'metrics':{},'contrasts':{},'judge':{}}
  for size in SIZES:
   summary[stage]['metrics'][size]={}
   for m in METRICS:
    x=np.array([r[m] for r in rows[size]])
    summary[stage]['metrics'][size][m]={'mean':float(x.mean()),'ci95':interval(x,ix)}
   for m in ['prompt_tokens','completion_tokens','retrieved_context_tokens']:
    x=np.array([r[m] for r in rows[size]])
    summary[stage]['metrics'][size][m]={'mean':float(x.mean()),'total':int(x.sum()),'min':int(x.min()),'max':int(x.max())}
  for m,sign in [('recall',1),('mrr',1),('f1',-1),('em',-1)]:
   x=sign*(np.array([r[m] for r in rows[128]])-np.array([r[m] for r in rows[1024]]))
   summary[stage]['contrasts'][m]={'direction':'128 - 1024' if sign==1 else '1024 - 128','mean':float(x.mean()),'ci95':interval(x,ix),'ci_simultaneous':interval(x,ix,.05/3),'paired_sd':float(x.std(ddof=1)),'planned_confirmation_simultaneous_halfwidth':float(2.394*x.std(ddof=1)/math.sqrt(600))}
  cc=[summary[stage]['contrasts'][m] for m in ['recall','mrr','f1']]
  verdict='PROVEN' if all(c['ci_simultaneous'][0]>0 and c['mean']>=.05 for c in cc) else 'DISPROVEN' if any(c['ci_simultaneous'][1]<0 for c in cc) else 'INCONCLUSIVE'
  summary[stage]['verdict']=verdict
  jj=records(OUT/f'{stage}_judge.jsonl')
  for size in SIZES:
   j=[r for r in jj if r['size']==size];valid=[r for r in j if r['judge_correct'] is not None]
   jix=np.random.default_rng(132).integers(len(valid),size=(10000,len(valid)))
   summary[stage]['judge'][size]={'ci95':interval(np.array([r['judge_correct'] for r in valid]),jix),'n':len(j),'valid':len(valid),'accuracy':float(np.mean([r['judge_correct'] for r in valid])),'agreement_f1_half':float(np.mean([r['judge_correct']==int(r['f1']>=.5) for r in valid])),'agreement_em':float(np.mean([r['judge_correct']==r['em'] for r in valid]))}
  valid=[r for r in jj if r['judge_correct'] is not None]
  summary[stage]['judge_total']={'n':len(jj),'valid':len(valid),'agreement_f1_half':float(np.mean([r['judge_correct']==int(r['f1']>=.5) for r in valid])),'agreement_em':float(np.mean([r['judge_correct']==r['em'] for r in valid])),'prompt_tokens':sum(r['prompt_tokens'] for r in jj),'completion_tokens':sum(r['completion_tokens'] for r in jj)}
  audit[stage]={'documents':len(data['documents']),'questions':n,'unique_question_ids':len(set(ids)),'dataset_info':data['info'],'sizes':{}}
  for size in SIZES:
   chunks=json.loads((OUT/f'{stage}_chunks_{size}.json').read_text());doc_chunk_counts={}
   for ch in chunks:doc_chunk_counts[ch['doc_id']]=doc_chunk_counts.get(ch['doc_id'],0)+1
   counts=np.array([c['n_tokens'] for c in chunks]);rs=rows[size]
   audit[stage]['sizes'][size]={'chunks':len(chunks),'single_chunk_documents_fraction':float(np.mean([doc_chunk_counts[d['id']]==1 for d in data['documents']])),'full_chunks_fraction':float(np.mean(counts==size)),'mean_chunk_tokens':float(counts.mean()),'max_chunk_tokens':int(counts.max()),'truncated_answers':sum(r['finish_reason']=='length' for r in rs),'unique_retrieved_documents_mean':float(np.mean([len(set(r['retrieved_doc_ids'])) for r in rs]))}
  a,b=rows[512],rows[1024]
  audit[stage]['identical_512_1024_top5_document_lists_fraction']=float(np.mean([r['retrieved_doc_ids']==s['retrieved_doc_ids'] for r,s in zip(a,b)]))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
 colors={'pilot':'#64748b','confirm':'#047d91'}
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for ax,m in zip(axes.flat,METRICS):
  for stage in ['pilot','confirm']:
   vals=[summary[stage]['metrics'][s][m] for s in SIZES];y=np.array([v['mean'] for v in vals])*100;lo=np.array([v['ci95'][0] for v in vals])*100;hi=np.array([v['ci95'][1] for v in vals])*100
   ax.errorbar(np.arange(4)+(-.035 if stage=='pilot' else .035),y,yerr=[y-lo,hi-y],marker='o',capsize=4,label=f'{"SQuAD pilot" if stage=="pilot" else "HotpotQA confirmation"} (n={summary[stage]["n"]})',color=colors[stage])
  ax.set(title=LABELS[m],xlabel='Chunk size (tokens)',ylabel='Score (%)');ax.set_xticks(range(4),SIZES);ax.grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Does the chunk-size winner change with the score?',fontsize=17);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(FIG/'metrics.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4.6));ms=['recall','mrr','f1'];y=np.arange(3)
 for off,stage in [(-.08,'pilot'),(.08,'confirm')]:
  cc=[summary[stage]['contrasts'][m] for m in ms];x=np.array([c['mean'] for c in cc])*100;lo=np.array([c['ci_simultaneous'][0] for c in cc])*100;hi=np.array([c['ci_simultaneous'][1] for c in cc])*100
  ax.errorbar(x,y+off,xerr=[x-lo,hi-x],fmt='o',capsize=4,color=colors[stage],label=stage.title())
 ax.axvline(0,color='black',lw=1);ax.axvline(5,color='#d97706',ls='--',label='Five-point practical threshold');ax.set_yticks(y,['Recall@5: 128 minus 1024','MRR@5: 128 minus 1024','Answer F1: 1024 minus 128']);ax.invert_yaxis();ax.set_xlabel('Paired difference (percentage points; positive supports hypothesis)');ax.set_title('Preregistered tests: familywise 95% confidence intervals');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=9,loc='best');fig.tight_layout();fig.savefig(FIG/'contrasts.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 for stage in ['pilot','confirm']:
  x=[summary[stage]['metrics'][s]['prompt_tokens']['mean'] for s in SIZES];y=[summary[stage]['metrics'][s]['f1']['mean']*100 for s in SIZES];ax.plot(x,y,'o-',color=colors[stage],label=stage.title())
  for a,b,s in zip(x,y,SIZES):
   offset=({128:(6,-1),256:(-5,35),512:(60,6),1024:(60,-22)}[s] if stage=='confirm' else (6,5))
   ax.annotate(str(s),(a,b),xytext=offset,textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':colors[stage],'lw':.7} if stage=='confirm' and s!=128 else None)
 ax.set(xlabel='Mean actual answer prompt tokens (including instructions)',ylabel='Mean answer token F1 (%)',title='Reading budget depends on the corpus');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(FIG/'tokens_vs_f1.png');plt.close(fig)
 def fmt(v):return f"{v['mean']*100:.1f} [{v['ci95'][0]*100:.1f}, {v['ci95'][1]*100:.1f}]"
 table_rows=[]
 for stage in ['pilot','confirm']:
  for s in SIZES:
   mm=summary[stage]['metrics'][s];table_rows.append(['Pilot' if stage=='pilot' else 'Confirm',str(s)]+[fmt(mm[m]) for m in METRICS]+[f"{mm['prompt_tokens']['mean']:,.0f}"])
 fig,ax=plt.subplots(figsize=(16,5));ax.axis('off');table=ax.table(cellText=table_rows,colLabels=['Dataset','Chunk\ntokens','Recall@5\n% [95% CI]','MRR@5\n% [95% CI]','Answer F1\n% [95% CI]','Exact match\n% [95% CI]','Mean prompt\ntokens'],loc='center',cellLoc='center',colWidths=[.09,.07,.17,.17,.17,.17,.1]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.05)
 for (r,c),cell in table.get_celld().items():
  cell.set_edgecolor('#e2e8f0')
  if r==0:cell.set_facecolor('#164e63');cell.set_text_props(color='white',weight='bold');cell.set_height(.14)
  else:cell.set_facecolor('#f0f9ff' if r>4 else ('#f8fafc' if r%2 else 'white'))
 ax.set_title('RAG chunk size: retrieval, answers and reading budget',fontsize=18,pad=20);fig.text(.5,.035,'SQuAD pilot n=300 | HotpotQA confirmation n=600 | Paired question bootstrap, 10,000 resamples | Qwen3-4B answers',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(FIG/'table.png',bbox_inches='tight');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
