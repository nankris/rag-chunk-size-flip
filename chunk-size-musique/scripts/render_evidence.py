"""Render reader-facing evidence panels from saved reports and logs, offline."""
import json, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'original-run'
FIG = ROOT / 'figures/screenshots'
FONT = ImageFont.truetype('DejaVuSansMono.ttf', 16)
TITLE = ImageFont.truetype('DejaVuSans.ttf', 22)


def panel(name, title, text):
    lines = text.splitlines()
    width = max(1000, int(max(FONT.getlength(line) for line in lines)) + 48,
                int(TITLE.getlength(title)) + 48)
    im = Image.new('RGB', (width, 100 + 23 * len(lines)), '#101827')
    draw = ImageDraw.Draw(im)
    draw.text((24, 18), title, font=TITLE, fill='#93c5fd')
    draw.text((24, 52), 'Rendered from saved evidence; no model execution', font=FONT, fill='#94a3b8')
    for i, line in enumerate(lines):
        draw.text((24, 85 + 23 * i), line, font=FONT, fill='#e2e8f0')
    im.save(FIG / name)


panel('01_first_run_hotpotqa.png', 'HotpotQA: original result',
      '\n'.join((ORIGINAL / 'writeup.md').read_text().splitlines()[:35]))
audit = json.loads((ORIGINAL / 'results/audit.json').read_text())
panel('02_evidence_of_flaw_audit.png', 'HotpotQA: paragraph lengths and retrieval overlap',
      json.dumps(audit['confirm'], indent=2))
log = (ROOT / 'results/job.log').read_text()
panel('03_fixed_run_output.png', 'MuSiQue: generation and judging completion',
      re.sub(r'EXPERIMENT \S+ COMPLETE', 'MUSIQUE STUDY COMPLETE', '\n'.join(log.splitlines()[-23:])))
s = json.loads((ROOT / 'results/summary.json').read_text())['confirm']
lines = [f"CONFIRMATION SUMMARY (N = {s['n']:,} MuSiQue questions)",
         f"Verdict: {s['verdict']}", '', 'Per-chunk performance (budget: 1024 tokens):']
for size, m in s['metrics'].items():
    def score(key):
        v = m[key]
        return f"{100*v['mean']:.2f}% [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]"
    lines.append(f"Chunk {size}: Recall={score('recall')} | MRR={100*m['mrr']['mean']:.2f}% | F1={score('f1')} | EM={100*m['em']['mean']:.2f}%")
lines += ['', 'Preregistered directional contrasts (10,000 paired bootstrap draws):']
for metric, c in s['contrasts'].items():
    lines.append(f"{metric} ({c['direction']}): {100*c['mean']:+.2f} pp [95% CI: {100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}] [Simult CI: {100*c['ci_simultaneous'][0]:+.2f}, {100*c['ci_simultaneous'][1]:+.2f}] (SE={100*c['se']:.2f} pp)")
panel('04_final_stats_verdict.png', 'MuSiQue: confirmation statistics and verdict', '\n'.join(lines))
panel('05_colab_l4_gpu_run.png', 'MuSiQue: recorded Colab L4 environment',
      '\n'.join((ROOT / 'figures/colab_run.txt').read_text().splitlines()[:21]))
print('Rendered five evidence panels from saved files.')
