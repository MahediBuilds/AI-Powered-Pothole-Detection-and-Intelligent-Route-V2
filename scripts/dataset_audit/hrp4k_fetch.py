"""Fetch HRP4K annotation files (+ optional sample images) via HTTP range. Usage: python -I hrp4k_fetch.py <out_dir> [images_list.json]
Writes bytes to out_dir/<path>; never executes anything."""
import sys, os, json, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import HttpRangeFile
URL = 'https://zenodo.org/api/records/21941931/files/HRP4K.zip/content'
out = sys.argv[1]
f = HttpRangeFile(URL, block=1024 * 1024)
z = zipfile.ZipFile(f)
if len(sys.argv) > 2:
    names = json.load(open(sys.argv[2]))
else:
    names = [i.filename for i in z.infolist() if i.filename.endswith('.json') or i.filename.endswith('.txt')]
for n in names:
    if '..' in n or n.startswith('/'): continue
    p = os.path.join(out, n); os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p): continue
    with open(p, 'wb') as g: g.write(z.read(n))
print('files', len(names), 'bytes_fetched', f.bytes_fetched, 'requests', f.requests)
