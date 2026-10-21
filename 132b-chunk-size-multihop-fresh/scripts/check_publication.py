"""Check publication completeness without changing reported results."""
import json,re,hashlib
from pathlib import Path
from PIL import Image
P=Path(__file__).resolve().parents[1]
required=['README.md','writeup.md','blog.md','figures/answer_f1.png','figures/effects_and_tokens.png','figures/table.png','figures/diagram.png','figures/diagram.excalidraw','figures/colab_run.txt','results/summary.json','results/validation.json','results/usage_final.json','results/timing.json','results/SESSION_STOPPED']
assert all((P/f).is_file() for f in required)
blog=(P/'blog.md').read_text();words=len(blog.split());assert 1200<=words<=1700,words
assert '—' not in blog
assert 'Code: <repo link placeholder>' in blog
for f in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',blog):assert (P/f).is_file(),f
assert json.loads((P/'results/validation.json').read_text())['passed']
u=json.loads((P/'results/usage_final.json').read_text());assert u['cu_upper_bound']<=6,u
s=json.loads((P/'results/summary.json').read_text());assert all(r['n']==3600 for r in s['arms'])
audit={'passed':True,'blog_words':words,'verdict':s['verdict'],'figures':{}}
for name in ['answer_f1.png','effects_and_tokens.png','table.png','diagram.png']:
    im=Image.open(P/'figures'/name);audit['figures'][name]={'pixels':list(im.size),'dpi':im.info.get('dpi')}
audit['sha256']={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in required if f!='results/SESSION_STOPPED'}
(P/'results/publication_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
