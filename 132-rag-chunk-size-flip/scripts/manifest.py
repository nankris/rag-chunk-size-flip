"""Record hashes for durable data, scripts, figures and prose artifacts."""
from pathlib import Path
import hashlib,json,datetime
root=Path(__file__).resolve().parents[1]
files=[p for sub in ['results','scripts','figures'] for p in (root/sub).rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='manifest.json']
files += [root/n for n in ['README.md','writeup.md','blog.md'] if (root/n).exists()]
records={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
(root/'results/manifest.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records},indent=2))
print('Hashed',len(records),'files')
