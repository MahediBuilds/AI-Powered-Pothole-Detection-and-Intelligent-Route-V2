#!/usr/bin/env python3
"""Static checks for the report sources (no LaTeX needed).

Usage:  python tools/check_tex.py

Checks main.tex and chapters/*.tex for: balanced braces, matching \\begin/\\end,
non-ASCII characters, dash characters and double hyphens in prose, labels that are
defined twice, \\ref/\\pageref targets without a label, and \\cite keys missing
from references.bib.  Exit status 1 if a hard error is found; double hyphens are
listed for manual review because mAP\\textsubscript{50--95} and the academic year are allowed.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip_comments(text):
    out = []
    for line in text.split("\n"):
        i = 0
        cut = None
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def main():
    files = [os.path.join(ROOT, "main.tex")] + sorted(glob.glob(os.path.join(ROOT, "chapters", "*.tex")))
    bib = open(os.path.join(ROOT, "references.bib"), encoding="utf-8").read()
    bib_keys = set(re.findall(r"^@\w+\{([^,\s]+),", bib, re.M))
    errors = 0
    labels = {}
    refs = []
    for path in files:
        if not os.path.exists(path):
            continue
        rel = os.path.relpath(path, ROOT)
        raw = open(path, encoding="utf-8").read()
        text = strip_comments(raw)
        # non-ASCII
        for n, line in enumerate(raw.split("\n"), 1):
            bad = [c for c in line if ord(c) > 127]
            if bad:
                print("%s:%d non-ASCII %r" % (rel, n, "".join(bad)))
                errors += 1
        # braces
        depth = 0
        for n, line in enumerate(text.split("\n"), 1):
            i = 0
            while i < len(line):
                c = line[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth < 0:
                        print("%s:%d extra closing brace" % (rel, n))
                        errors += 1
                        depth = 0
                i += 1
        if depth != 0:
            print("%s: unbalanced braces (depth %d at end)" % (rel, depth))
            errors += 1
        # environments
        stack = []
        for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", text):
            n = text.count("\n", 0, m.start()) + 1
            if m.group(1) == "begin":
                stack.append((m.group(2), n))
            else:
                if not stack or stack[-1][0] != m.group(2):
                    print("%s:%d \\end{%s} does not match %s" % (rel, n, m.group(2), stack[-1] if stack else None))
                    errors += 1
                    if stack:
                        stack.pop()
                else:
                    stack.pop()
        for name, n in stack:
            print("%s:%d \\begin{%s} never closed" % (rel, n, name))
            errors += 1
        # dashes
        for n, line in enumerate(text.split("\n"), 1):
            if re.search(r"---", line):
                print("%s:%d triple hyphen (em dash)" % (rel, n))
                errors += 1
            for m in re.finditer(r"--", line):
                ctx = line[max(0, m.start() - 25):m.end() + 12]
                print("%s:%d double hyphen (review): ...%s..." % (rel, n, ctx))
        # labels, refs, cites
        for m in re.finditer(r"\\label\{([^}]*)\}", text):
            if m.group(1) in labels:
                print("%s: label %s also defined in %s" % (rel, m.group(1), labels[m.group(1)]))
                errors += 1
            labels[m.group(1)] = rel
        for m in re.finditer(r"\\(?:ref|pageref|eqref)\{([^}]*)\}", text):
            refs.append((m.group(1), rel, text.count("\n", 0, m.start()) + 1))
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])*\{([^}]*)\}", text):
            for key in m.group(1).split(","):
                key = key.strip()
                if key and key not in bib_keys:
                    print("%s:%d cite key not in references.bib: %s" % (rel, text.count("\n", 0, m.start()) + 1, key))
                    errors += 1
    for key, rel, n in refs:
        if key not in labels:
            print("%s:%d \\ref to undefined label %s" % (rel, n, key))
            errors += 1
    print("files checked: %d, labels: %d, refs: %d, hard errors: %d" % (len(files), len(labels), len(refs), errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
