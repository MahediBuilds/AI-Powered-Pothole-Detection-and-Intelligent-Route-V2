"""Check that the verbatim excerpts in ch5_implementation.tex appear unchanged in the source files."""
import re, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = r"<PROJECT_ROOT>/New Implementation\src\roadsense"
tex = open(os.path.join(root, "chapters", "ch5_implementation.tex"), encoding="utf-8").read()
blocks = re.findall(r"\\begin\{verbatim\}\n(.*?)\\end\{verbatim\}", tex, flags=re.S)
files = ["detection/detector.py", "tracking/tracker.py", "routing/router.py"]
for b, f in zip(blocks, files):
    s = open(os.path.join(src, f), encoding="utf-8").read().replace("\r\n", "\n")
    print(f, "lines:", len(b.rstrip("\n").split("\n")), "exact match:", b in s, "max width:", max(len(l) for l in b.split("\n")))
