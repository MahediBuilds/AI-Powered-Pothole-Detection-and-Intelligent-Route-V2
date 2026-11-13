"""Mechanical checks for ch4/ch5 tex files (labels, refs, cites, figures, special characters, dashes)."""
import re, sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
files = [os.path.join(root, "chapters", n) for n in ("ch4_design.tex", "ch5_implementation.tex")]
bib = open(os.path.join(root, "references.bib"), encoding="utf-8").read()
keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
text = {f: open(f, encoding="utf-8").read() for f in files}
alltext = "\n".join(text.values())
labels = re.findall(r"\\label\{([^}]+)\}", alltext)
refs = re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", alltext)
print("labels:", len(labels), "dups:", [l for l in set(labels) if labels.count(l) > 1])
ext = {"ch:results", "ch:conclusion", "ch:intro", "ch:literature", "ch:requirements"}
print("undefined refs:", sorted(set(r for r in refs if r not in labels and r not in ext)))
print("external refs used:", sorted(set(r for r in refs if r in ext)))
cites = []
for m in re.finditer(r"\\cite\{([^}]+)\}", alltext):
    cites += [k.strip() for k in m.group(1).split(",")]
print("cites:", len(set(cites)))
print("missing cite keys:", sorted(set(c for c in cites if c not in keys)))
order = []
for c in cites:
    if c not in order:
        order.append(c)
print("first-citation order:", order)
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", alltext)
for f in figs:
    p = os.path.join(root, "figures", f)
    print(("ok  " if os.path.exists(p) else "MISSING ") + f)
for f, t in text.items():
    n = os.path.basename(f)
    print("==", n)
    print("non-ascii:", [c for c in set(t) if ord(c) > 127])
    print("em/en dash chars or '--' outside mAP:", len(re.findall(r"--", re.sub(r"mAP\\textsubscript\{50--95\}", "", t))))
    # unescaped special characters (outside verbatim)
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", t, flags=re.S)
    for name, pat in (("percent", r"(?<!\\)%"), ("hash", r"(?<!\\)#"), ("underscore outside texttt/math", None)):
        if pat:
            print(name, len(re.findall(pat, body)))
    # underscores outside \texttt{...} and math
    tmp = re.sub(r"\\texttt\{[^}]*\}", "", body)
    tmp = re.sub(r"\$[^$]*\$", "", tmp)
    tmp = re.sub(r"\\(label|ref|includegraphics)(\[[^\]]*\])?\{[^}]*\}", "", tmp)
    tmp = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", tmp, flags=re.S)
    print("bare underscores:", [tmp[max(0, m.start() - 25):m.end() + 10] for m in re.finditer(r"_", tmp)][:10])
    print("begin/end balance:", {e: (len(re.findall(r"\\begin\{%s\}" % e, t)), len(re.findall(r"\\end\{%s\}" % e, t))) for e in set(re.findall(r"\\begin\{(\w+\*?)\}", t))})
    print("braces:", t.count("{") - t.count("}"), "(verbatim contents included)")
    words = len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[a-zA-Z]+", " ", body)))
    print("approx words:", words)
