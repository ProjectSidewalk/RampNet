"""A seekable read-only file over HTTP range requests.

Lets ``zipfile`` read a remote zip's central directory and a few named members without
downloading the archive -- how the crop-cutter validation reads individual crops out of the
30 GB HF ``sidewalk-tagger-ai-validated`` zip. Same technique as ``_RangeFile`` in
``scripts/analysis/ps_supervision_audit.py`` (PR #175); duplicated here so this PR does not
depend on that one merging first.

Usage::

    import io, zipfile
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20))
    data = z.read("CurbRamp/crops/gsv-seattle-9-CurbRamp.png")
"""
import io

import requests

UA = {"User-Agent": "Mozilla/5.0 (RampNet research)"}


class RangeFile(io.RawIOBase):
    def __init__(self, url, session=None):
        self.s = session or requests.Session()
        self.url = self.s.head(url, headers=UA, allow_redirects=True).url
        self.size = int(self.s.head(self.url, headers=UA).headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n <= 0 or self.pos >= self.size:
            return 0
        r = self.s.get(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
        r.raise_for_status()
        if r.status_code != 206:
            # a server that ignores Range sends the whole body with 200; never splice that in
            raise IOError(f"{self.url}: expected 206 Partial Content for a range read, got {r.status_code}")
        if len(r.content) > n:
            raise IOError(f"{self.url}: range read returned {len(r.content)} bytes for {n} requested")
        b[:len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)
