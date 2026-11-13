"""Seekable HTTP-Range file wrapper for zipfile.ZipFile, with block cache and transfer accounting."""
import io, time, requests

class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=256 * 1024, log=None):
        self.url = url; self.block = block; self.pos = 0; self.cache = {}; self.bytes_fetched = 0; self.requests = 0
        self.s = requests.Session(); self.s.headers['User-Agent'] = 'dataset-audit/1.0'
        r = self._get(0, 0); self.size = int(r.headers['Content-Range'].split('/')[1])
    def _get(self, a, b):
        for attempt in range(6):
            r = self.s.get(self.url, headers={'Range': f'bytes={a}-{b}'}, timeout=120, allow_redirects=True)
            if r.status_code == 206:
                self.requests += 1; self.bytes_fetched += len(r.content); return r
            if r.status_code == 429:
                time.sleep(int(r.headers.get('retry-after', 30)) + 1); continue
            if r.status_code == 200:
                raise RuntimeError('server ignored Range')
            time.sleep(2)
        raise RuntimeError(f'range fail {r.status_code}')
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block; b = min(a + self.block, self.size) - 1
            self.cache[i] = self._get(a, b).content
            if len(self.cache) > 64: self.cache.pop(next(iter(self.cache)))
        return self.cache[i]
    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size: n = self.size - self.pos
        if n <= 0: return b''
        # large reads: fetch directly in one request
        if n > 2 * self.block:
            data = self._get(self.pos, self.pos + n - 1).content
            self.pos += n; return data
        out = bytearray(); end = self.pos + n
        while self.pos < end:
            i = self.pos // self.block; blk = self._blk(i); o = self.pos - i * self.block
            take = blk[o:o + (end - self.pos)]; out += take; self.pos += len(take)
            if not take: break
        return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
