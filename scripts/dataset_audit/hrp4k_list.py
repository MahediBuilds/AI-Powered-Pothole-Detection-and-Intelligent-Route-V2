"""List HRP4K zip central directory remotely. Usage: python -I hrp4k_list.py <out_json>"""
import sys, os, json, zipfile, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import HttpRangeFile
URL = 'https://zenodo.org/api/records/21941931/files/HRP4K.zip/content'
f = HttpRangeFile(URL, block=1024 * 1024)
z = zipfile.ZipFile(f)
infos = z.infolist()
L = [{'name': i.filename, 'size': i.file_size, 'csize': i.compress_size, 'method': i.compress_type, 'offset': i.header_offset, 'date': list(i.date_time)} for i in infos]
ext = collections.Counter(os.path.splitext(i.filename)[1].lower() for i in infos if not i.is_dir())
top = collections.Counter('/'.join(i.filename.split('/')[:2]) for i in infos)
R = {'zip_size': f.size, 'n_entries': len(infos), 'ext_counts': ext, 'bytes_fetched': f.bytes_fetched, 'requests': f.requests, 'entries': L}
json.dump(R, open(sys.argv[1], 'w'))
print(f.size, len(infos), ext, f.bytes_fetched, f.requests)
for k, v in sorted(top.items())[:60]: print(k, v)
