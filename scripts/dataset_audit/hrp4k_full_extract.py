"""Safe extraction of HRP4K.zip: only HRP4K/{train,valid,test}/{images/*.jpg,labels/*.txt} and HRP4K/*.json,
rejects any other member name / path traversal. usage: hrp4k_full_extract.py ZIP DEST"""
import sys, zipfile, os, re

zp, dest = sys.argv[1], sys.argv[2]
ok = re.compile(r"^HRP4K/((train|valid|test)/(images/\d+\.jpg|labels/\d+\.txt)|[A-Za-z0-9_]+\.json)$")
z = zipfile.ZipFile(zp)
bad = [i.filename for i in z.infolist() if not i.is_dir() and not ok.match(i.filename)]
print("members", len(z.infolist()), "unexpected", bad[:20], len(bad))
print("testzip (CRC of every member):", z.testzip())
os.makedirs(dest, exist_ok=True)
n = 0
for i in z.infolist():
    if i.is_dir() or not ok.match(i.filename):
        continue
    t = os.path.join(dest, *i.filename.split("/"))
    assert os.path.abspath(t).startswith(os.path.abspath(dest) + os.sep)
    os.makedirs(os.path.dirname(t), exist_ok=True)
    with z.open(i) as s, open(t, "wb") as o:
        while True:
            b = s.read(1 << 22)
            if not b:
                break
            o.write(b)
    n += 1
print("extracted", n)
