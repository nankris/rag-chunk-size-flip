"""Publication figures from the saved analysis, no GPU required."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];S=json.loads((P/'results/summary.json').read_text());F=P/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.titlesize':15,'axes.labelsize':12,'figure.dpi':150,'savefig.dpi':150,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
colors=['#2563eb','#d97706']
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for i,reg in enumerate(['budget','equal_k']):
 rows=[r for r in S['arms'] if r['regime']==reg];ax=axs[i];v=np.array([r['f1']*100 for r in rows]);lo=np.array([r['f1_lo']*100 for r in rows]);hi=np.array([r['f1_hi']*100 for r in rows])
 ax.errorbar(range(4),v,yerr=[v-lo,hi-v],fmt='o-',capsize=5,color=colors[i],lw=2,ms=7)
 for x,y in enumerate(v):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=10)
 ax.set_xticks(range(4),[128,256,512,1024]);ax.set_xlabel('Maximum chunk size (tokens)');ax.set_ylabel('Answer token F1 (%)');ax.set_title('Fixed 1,024 evidence tokens' if reg=='budget' else 'Equal top-5 chunks');ax.grid(axis='y',alpha=.2);ax.set_ylim(24,32)
fig.suptitle('Larger chunk limits under two reading policies',fontsize=17);fig.text(.5,.015,'3,600 fresh 2WikiMultiHopQA questions · Qwen3-8B · marginal 95% paired-bootstrap intervals',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(F/'answer_f1.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,5))
for i,reg in enumerate(['budget','equal_k']):
 cs=[r for r in S['contrasts'] if r['regime']==reg];y=np.arange(3)+(i-.5)*.15;v=np.array([r['gain_pp'] for r in cs]);lo=np.array([r['simultaneous_lo_pp'] for r in cs]);hi=np.array([r['simultaneous_hi_pp'] for r in cs])
 axs[0].errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',capsize=4,color=colors[i],label='Fixed token budget' if i==0 else 'Equal top-5')
axs[0].axvline(0,color='#64748b',lw=1);axs[0].axvline(1.5,color='#b91c1c',ls='--',lw=1,label='Practical margin: +1.5 pp');axs[0].set_yticks(range(3),['256 − 128','512 − 128','1024 − 128']);axs[0].set_xlabel('Paired answer F1 difference (percentage points)');axs[0].set_title('Six simultaneous contrasts');axs[0].set_ylim(-.35,3.1);axs[0].legend(fontsize=9,loc='upper right');axs[0].grid(axis='x',alpha=.2)
for i,reg in enumerate(['budget','equal_k']):
 rows=[r for r in S['arms'] if r['regime']==reg];axs[1].plot([128,256,512,1024],[r['prompt_tokens'] for r in rows],'o-',color=colors[i],label='Fixed token budget' if i==0 else 'Equal top-5')
axs[1].set_xscale('log',base=2);axs[1].set_xticks([128,256,512,1024],[128,256,512,1024]);axs[1].set_xlabel('Maximum chunk size (tokens)');axs[1].set_ylabel('Mean served-model prompt tokens');axs[1].set_title('Actual reader inputs');axs[1].grid(axis='y',alpha=.2);axs[1].legend(fontsize=9)
fig.suptitle('The effect and the token-budget check',fontsize=17);fig.text(.5,.01,'Intervals: nominal 95% familywise target across six F1 contrasts; 20,000 paired bootstrap draws',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(F/'effects_and_tokens.png');plt.close(fig)
cols=['Policy','Chunk\nlimit','F1 %\n[95% CI]','EM %\n[95% CI]','Fact recall %\n[95% CI]','Mean input\ntokens','Mean k'];cells=[]
for r in S['arms']:
 ci=lambda m:f'{100*r[m]:.2f}\n[{100*r[m+"_lo"]:.2f}, {100*r[m+"_hi"]:.2f}]'
 cells.append(['1,024 tokens' if r['regime']=='budget' else 'Top 5',str(r['size']),ci('f1'),ci('em'),ci('support_recall'),f'{r["prompt_tokens"]:,.0f}',f'{r["k"]:.1f}'])
fig,ax=plt.subplots(figsize=(13,7));ax.axis('off');t=ax.table(cellText=cells,colLabels=cols,cellLoc='center',loc='center',colWidths=[.12,.08,.2,.2,.2,.12,.08]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.95)
for (row,col),cell in t.get_celld().items():
 cell.set_edgecolor('#cbd5e1')
 if row==0:cell.set_facecolor('#172554');cell.set_text_props(color='white',weight='bold')
 else:cell.set_facecolor('#eff6ff' if row<=4 else '#fff7ed')
ax.set_title('Chunk-size results on 3,600 fresh multi-hop questions',fontsize=18,pad=18);fig.text(.5,.025,'Qwen3-8B on L4 · Fact recall requires the full supporting sentence · Marginal 95% bootstrap intervals',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(F/'table.png');plt.close(fig)
print('WROTE 3 publication charts/table')
