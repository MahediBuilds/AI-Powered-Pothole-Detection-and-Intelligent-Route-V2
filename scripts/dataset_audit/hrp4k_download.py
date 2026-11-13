"""Parallel, resumable range download of the HRP4K v1.01 zip (Zenodo record 21941931).

Chunks go to <out>.parts/NNNN.part (resumable per chunk) and are concatenated into <out>
at the end. Writes only under data/external/hrp4k/.
usage: hrp4k_download.py OUT_ZIP [workers]
"""
import os, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "https://zenodo.org/api/records/21941931/files/HRP4K.zip/content"
SIZE = 12795844822
CH = 32 * 1024 * 1024
out = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
parts = out + ".parts"; os.makedirs(parts, exist_ok=True)
n = (SIZE + CH - 1) // CH
done_bytes = [0]; lock = threading.Lock()

def fetch(i):
    lo = i * CH; hi = min(SIZE, lo + CH) - 1; need = hi - lo + 1
    p = os.path.join(parts, f"{i:05d}.part")
    for attempt in range(200):
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if have == need:
            return
        if have > need:
            os.remove(p); have = 0
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={lo+have}-{hi}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, "ab") as f:
                while True:
                    b = r.read(1 << 18)
                    if not b: break
                    f.write(b)
                    with lock: done_bytes[0] += len(b)
        except Exception as e:
            time.sleep(3)
    raise RuntimeError(f"chunk {i} failed")

def report():
    t0 = time.time()
    while not stop.is_set():
        time.sleep(60)
        print(f"{time.time()-t0:.0f}s {done_bytes[0]/1e9:.2f} GB this run", flush=True)
stop = threading.Event(); threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(workers) as ex:
    list(ex.map(fetch, range(n)))
stop.set()
with open(out, "wb") as o:
    for i in range(n):
        with open(os.path.join(parts, f"{i:05d}.part"), "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b: break
                o.write(b)
print("assembled", os.path.getsize(out), flush=True)
