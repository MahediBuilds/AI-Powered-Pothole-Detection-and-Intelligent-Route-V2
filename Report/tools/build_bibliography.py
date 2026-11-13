#!/usr/bin/env python3
"""Build references_generated.tex from the keys cited in the report.

Usage (from anywhere):  python tools/build_bibliography.py

What it does
  1. Reads main.tex and follows its \\input{...} lines recursively, so the text is
     scanned in the order LaTeX will read it.  Files in chapters/*.tex that main.tex
     never inputs are scanned afterwards (alphabetical order) and reported as such.
  2. Collects every key used in \\cite{...} (also \\citep, \\citet, \\cite[..]{..}),
     in order of first appearance.  LaTeX comments (unescaped %) are ignored.
  3. Looks each key up in references.bib.  A missing key is a hard failure: nothing is
     written and the exit status is 1.
  4. Writes references_generated.tex: a thebibliography environment with one
     \\bibitem per cited work, in order of first citation, in one numeric style
     (IEEE-like: authors, "title," venue in italics, vol., no., pp., year, DOI or URL).
  5. Writes _working/citation_report.txt: cited keys in order, uncited keys (left out
     of the list), missing keys, and keys that only occur in files main.tex does not input.

Only the Python 3.10 standard library is used.  The bib `note` and `keywords`
fields are working notes and are never printed.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "main.tex")
BIB = os.path.join(ROOT, "references.bib")
OUT = os.path.join(ROOT, "references_generated.tex")
REPORT = os.path.join(ROOT, "_working", "citation_report.txt")
CHAPTER_DIR = os.path.join(ROOT, "chapters")

CITE_PAT = r"\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}"
INPUT_PAT = r"\\(?:input|include)\s*\{([^}]*)\}"


class BuildError(Exception):
    pass


# ----------------------------------------------------------------------------
# TeX scanning
# ----------------------------------------------------------------------------
def strip_comments(text):
    """Remove unescaped % comments, keeping line structure."""
    out = []
    for line in text.split("\n"):
        i = 0
        cut = None
        while i < len(line):
            c = line[i]
            if c == "\\":
                i += 2
                continue
            if c == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def read_tex(path):
    with open(path, "r", encoding="utf-8") as fh:
        return strip_comments(fh.read())


def resolve_tex(name):
    name = name.strip()
    cand = os.path.join(ROOT, name)
    if not cand.lower().endswith(".tex"):
        cand += ".tex"
    return cand


def scan_file(path, seen, events, depth=0):
    """Walk a file; append (key, file, line) events in reading order."""
    if depth > 10:
        raise BuildError("\\input nesting deeper than 10 levels at " + path)
    seen.add(os.path.normcase(os.path.abspath(path)))
    text = read_tex(path)
    token_re = re.compile(CITE_PAT + "|" + INPUT_PAT)
    rel = os.path.relpath(path, ROOT)
    for m in token_re.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        if m.group(1) is not None:
            for key in m.group(1).split(","):
                key = key.strip()
                if key:
                    events.append((key, rel, line_no))
        else:
            target = resolve_tex(m.group(2))
            if os.path.basename(target) == "references_generated.tex":
                continue
            if not os.path.exists(target):
                # Chapter files from other writers may not exist yet.
                events.append(("__missing_file__", rel + " -> " + m.group(2), line_no))
                continue
            scan_file(target, seen, events, depth + 1)


# ----------------------------------------------------------------------------
# BibTeX parsing
# ----------------------------------------------------------------------------
def parse_bib(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    entries = {}
    order = []
    i = 0
    n = len(text)
    head = re.compile(r"@([A-Za-z]+)\s*([{(])")
    while i < n:
        c = text[i]
        if c == "%":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c != "@":
            i += 1
            continue
        m = head.match(text, i)
        if not m:
            i += 1
            continue
        etype = m.group(1).lower()
        j = m.end()
        depth = 1
        k = j
        while k < n and depth > 0:
            ch = text[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth != 0:
            raise BuildError("Unbalanced braces in a bib entry starting at: " + text[i:i + 60])
        body = text[j:k - 1]
        i = k
        if etype in ("comment", "preamble", "string"):
            continue
        km = re.match(r"\s*([^,\s]+)\s*,", body)
        if not km:
            raise BuildError("Cannot read entry key near: " + body[:60])
        key = km.group(1)
        if key in entries:
            raise BuildError("Duplicate bib key: " + key)
        fields = parse_fields(body[km.end():], key)
        entries[key] = {"type": etype, "fields": fields}
        order.append(key)
    return entries, order


def parse_fields(s, key):
    fields = {}
    i = 0
    n = len(s)
    name_re = re.compile(r"\s*,?\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*")
    bare_re = re.compile(r"[^,\s}]+")
    while i < n:
        m = name_re.match(s, i)
        if not m:
            break
        name = m.group(1).lower()
        i = m.end()
        if i >= n:
            break
        if s[i] == "{":
            depth = 0
            j = i
            while j < n:
                ch = s[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                raise BuildError("Unbalanced braces in field %s of %s" % (name, key))
            value = s[i + 1:j]
            i = j + 1
        elif s[i] == '"':
            j = i + 1
            while j < n and s[j] != '"':
                if s[j] == "\\":
                    j += 1
                j += 1
            value = s[i + 1:j]
            i = j + 1
        else:
            m2 = bare_re.match(s, i)
            value = m2.group(0)
            i = m2.end()
        fields[name] = re.sub(r"\s+", " ", value).strip()
    return fields


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------
UNICODE_MAP = {
    "\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
    "\u201c": "``", "\u201d": "''", "\u00e9": "{\\'e}", "\u00e8": "{\\`e}",
    "\u00e4": '{\\"a}', "\u00f6": '{\\"o}', "\u00fc": '{\\"u}', "\u00e1": "{\\'a}",
    "\u00ed": "{\\'\\i}", "\u00f3": "{\\'o}", "\u00fa": "{\\'u}", "\u00e7": "{\\c c}",
    "\u00f1": "{\\~n}", "\u00a0": "~",
}


def deunicode(s):
    return "".join(UNICODE_MAP.get(ch, ch) for ch in s)


def split_top_level(s, sep_pattern):
    """Split s on a regex separator, only at brace depth 0."""
    parts = []
    depth = 0
    start = 0
    i = 0
    n = len(s)
    sep = re.compile(sep_pattern)
    while i < n:
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            m = sep.match(s, i)
            if m and m.end() > i:
                parts.append(s[start:i])
                start = m.end()
                i = m.end()
                continue
        i += 1
    parts.append(s[start:])
    return [p.strip() for p in parts if p.strip()]


def first_char_unit(token):
    """First letter of a name token, keeping a leading accent group intact."""
    if token.startswith("{"):
        depth = 0
        for idx, ch in enumerate(token):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return token[:idx + 1]
        return token
    return token[0].upper()


def initials(given):
    out = []
    for word in split_top_level(given, r"\s+"):
        pieces = split_top_level(word, r"-")
        subs = []
        for p in pieces:
            if re.match(r"^[A-Z]\.$", p):
                subs.append(p)  # already an initial such as "E."
            else:
                subs.append(first_char_unit(p) + ".")
        out.append("-".join(subs))
    return " ".join(out)


def is_fully_wrapped(name):
    if not (name.startswith("{") and name.endswith("}")):
        return False
    depth = 0
    for idx, ch in enumerate(name):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and idx != len(name) - 1:
                return False
    return True


def format_person(name):
    name = name.strip()
    if is_fully_wrapped(name):  # corporate author, e.g. {{Ministry of ...}}
        inner = name[1:-1].strip()
        if is_fully_wrapped(inner):
            inner = inner[1:-1].strip()
        return inner
    comma_parts = split_top_level(name, r",")
    if len(comma_parts) >= 2:
        last = comma_parts[0]
        given = ", ".join(comma_parts[1:])
    else:
        words = split_top_level(name, r"\s+")
        last = words[-1]
        given = " ".join(words[:-1])
    ini = initials(given) if given else ""
    return (ini + " " + last).strip()


def format_authors(raw):
    names = split_top_level(raw, r"\s+and\s+")
    people = [format_person(n) for n in names]
    if len(people) > 6:
        return people[0] + " et al."
    if len(people) == 1:
        return people[0]
    if len(people) == 2:
        return people[0] + " and " + people[1]
    return ", ".join(people[:-1]) + ", and " + people[-1]


def page_text(pages, article_number_ok):
    if not pages:
        return ""
    if "-" in pages:
        return "pp. " + re.sub(r"-+", "--", pages)
    return ("Art. no. " if article_number_ok else "p. ") + pages


def quoted_title(tt):
    if tt.rstrip()[-1] in "?!":
        return "``%s''" % tt
    return "``%s,''" % tt


def format_entry(key, e):
    t = e["type"]
    f = {k: deunicode(v) for k, v in e["fields"].items()}
    if "author" not in f:
        raise BuildError("Entry %s has no author" % key)
    if "title" not in f:
        raise BuildError("Entry %s has no title" % key)
    authors = format_authors(f["author"])
    title = f["title"]
    year = f.get("year", "")
    parts = []
    tail_link = ""
    if "doi" in f:
        tail_link = "doi: \\url{https://doi.org/%s}" % f["doi"]
    elif "url" in f:
        tail_link = "\\url{%s}" % f["url"]

    if t == "article":
        parts.append(authors + ",")
        parts.append(quoted_title(title))
        parts.append("\\textit{%s}," % f.get("journal", ""))
        bits = []
        if "volume" in f:
            bits.append("vol. " + f["volume"])
        if "number" in f:
            bits.append("no. " + f["number"])
        if "pages" in f:
            bits.append(page_text(f["pages"], True))
        if year:
            bits.append(year)
        parts.append(", ".join(bits) + ".")
    elif t == "inproceedings":
        parts.append(authors + ",")
        parts.append(quoted_title(title))
        parts.append("in \\textit{%s}," % f.get("booktitle", ""))
        bits = []
        if "volume" in f:
            bits.append("vol. " + f["volume"])
        if year:
            bits.append(year)
        if "pages" in f:
            bits.append(page_text(f["pages"], False))
        parts.append(", ".join(bits) + ".")
    elif t == "book":
        parts.append(authors + ",")
        parts.append("\\textit{%s}," % title)
        bits = [b for b in (f.get("publisher", ""), year) if b]
        parts.append(", ".join(bits) + ".")
    elif t in ("techreport", "report"):
        parts.append(authors + ",")
        parts.append("\\textit{%s}," % title)
        inst = f.get("institution", "")
        if inst and inst in authors:
            inst = ""
        bits = [inst, f.get("address", ""), year]
        parts.append(", ".join(b for b in bits if b) + ".")
    elif t == "misc":
        parts.append(authors + ",")
        parts.append("\\textit{%s}," % title)
        bits = []
        hp = f.get("howpublished", "")
        if "eprint" in f:
            hp = (hp + " " if hp.lower().startswith("arxiv") else (hp + ", " if hp else "")) + "arXiv:" + f["eprint"]
        if hp:
            bits.append(hp)
        if year:
            bits.append(year)
        parts.append(", ".join(bits) + ".")
    else:
        raise BuildError("Entry %s has unsupported type %s" % (key, t))
    if tail_link:
        parts.append(tail_link + ".")
    text = " ".join(parts)
    check_text(text, key)
    return text


def check_text(text, key):
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise BuildError("Non-ASCII character in entry %s: %r" % (key, text[exc.start:exc.end + 5]))
    plain = re.sub(r"\\url\{[^}]*\}", "", text)
    for ch in "&%#_":
        if re.search(r"(?<!\\)" + re.escape(ch), plain):
            raise BuildError("Unescaped %r in entry %s: %s" % (ch, key, plain))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not os.path.exists(MAIN):
        raise BuildError("main.tex not found at " + MAIN)
    entries, bib_order = parse_bib(BIB)

    seen_files = set()
    events = []
    scan_file(MAIN, seen_files, events)

    extra_events = []
    if os.path.isdir(CHAPTER_DIR):
        for name in sorted(os.listdir(CHAPTER_DIR)):
            if not name.lower().endswith(".tex"):
                continue
            p = os.path.join(CHAPTER_DIR, name)
            if os.path.normcase(os.path.abspath(p)) in seen_files:
                continue
            scan_file(p, seen_files, extra_events)

    missing_files = [e for e in events + extra_events if e[0] == "__missing_file__"]
    events = [e for e in events if e[0] != "__missing_file__"]
    extra_events = [e for e in extra_events if e[0] != "__missing_file__"]

    cited = []
    first_seen = {}
    only_extra = []
    for key, rel, line in events:
        if key not in first_seen:
            first_seen[key] = (rel, line)
            cited.append(key)
    for key, rel, line in extra_events:
        if key not in first_seen:
            first_seen[key] = (rel, line)
            cited.append(key)
            only_extra.append(key)

    missing = [k for k in cited if k not in entries]
    uncited = [k for k in bib_order if k not in first_seen]

    lines = []
    lines.append("CITATION REPORT (written by tools/build_bibliography.py)")
    lines.append("")
    lines.append("Cited keys in order of first appearance: %d" % len(cited))
    for idx, k in enumerate(cited, 1):
        flag = ""
        if k in missing:
            flag = "   <-- MISSING FROM references.bib"
        elif k in only_extra:
            flag = "   (only in a chapter file that main.tex does not input)"
        lines.append("  [%d] %s   first at %s:%d%s" % (idx, k, first_seen[k][0], first_seen[k][1], flag))
    lines.append("")
    lines.append("Missing keys: %d" % len(missing))
    for k in missing:
        lines.append("  %s  first at %s:%d" % (k, first_seen[k][0], first_seen[k][1]))
    lines.append("")
    lines.append("Bib entries never cited (left out of the list): %d" % len(uncited))
    for k in uncited:
        tier = entries[k]["fields"].get("keywords", "")
        lines.append("  %s   [%s]" % (k, tier))
    lines.append("")
    lines.append("\\input targets not found (chapter files not written yet?): %d" % len(missing_files))
    for _k, what, line in missing_files:
        lines.append("  %s (line %d)" % (what, line))
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    if missing:
        sys.stderr.write("ERROR: cited keys missing from references.bib:\n")
        for k in missing:
            sys.stderr.write("  %s (first at %s:%d)\n" % (k, first_seen[k][0], first_seen[k][1]))
        sys.stderr.write("references_generated.tex was NOT written.\n")
        return 1

    items = []
    for k in cited:
        items.append("\\bibitem{%s} %s" % (k, format_entry(k, entries[k])))
    out = []
    out.append("% GENERATED by tools/build_bibliography.py from references.bib. Do not edit by hand.")
    out.append("%% %d works, in order of first citation." % len(cited))
    out.append("\\begin{thebibliography}{99}")
    out.append("\\addcontentsline{toc}{chapter}{REFERENCES}")
    out.extend(items)
    out.append("\\end{thebibliography}")
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")

    print("Cited: %d   uncited: %d   missing: %d" % (len(cited), len(uncited), len(missing)))
    if len(cited) > 99:
        print("WARNING: more than 99 references; widen the thebibliography label.")
    if missing_files:
        print("Note: %d \\input target(s) not found yet (see citation report)." % len(missing_files))
    print("Wrote " + OUT)
    print("Wrote " + REPORT)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        sys.stderr.write("ERROR: %s\n" % exc)
        sys.exit(1)
