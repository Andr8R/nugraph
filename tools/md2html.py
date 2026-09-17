#!/usr/bin/env python3
"""md2html.py -- render a workspace Markdown report as a standalone, shareable HTML page.

    python3 tools/md2html.py reports/<name>.md [reports/<name>.html]

Standard library only (GPVM has no markdown/pandoc). Supports the subset the reports use:
# title (+ italic subtitle paragraph), ## sections (rendered as cards), ### subsections
(Measured / Interpretation / Caveats / Setup get coloured badges), paragraphs, nested - and 1.
lists, pipe tables, fenced code (diff lines coloured), `code`, **bold**, *italic*, [links](url),
and --- separators (dropped; sections already separate). Output is deterministic: rebuilding
from the same Markdown gives the same HTML (the footer records the source sha256).
"""
import hashlib
import html
import re
import sys
from pathlib import Path

BADGES = {"measured": "b-ok", "setup": "b-run", "interpretation": "b-run",
          "caveats": "b-hyp", "how to read": "b-run"}

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1a1a1a;--muted:#5f6368;--line:#e3e3e0;--code:#f1f1ee;
  --ok:#0f7b4f;--ok-bg:#e6f4ec;--hyp:#8a5a00;--hyp-bg:#fdf3e0;--run:#1c4f8f;--run-bg:#e8f0fa;
  --add:#0f7b4f;--del:#a32020;--hl:#fffbe6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#15161a;--card:#1d1f24;--ink:#eceef2;--muted:#a2a8b4;--line:#2f333b;--code:#262930;
  --ok:#5fd39b;--ok-bg:#12332a;--hyp:#e5b45a;--hyp-bg:#33290f;--run:#7fb2f0;--run-bg:#13233a;
  --add:#5fd39b;--del:#f08a8a;--hl:#2a2616}}
:root[data-theme="dark"]{
  --bg:#15161a;--card:#1d1f24;--ink:#eceef2;--muted:#a2a8b4;--line:#2f333b;--code:#262930;
  --ok:#5fd39b;--ok-bg:#12332a;--hyp:#e5b45a;--hyp-bg:#33290f;--run:#7fb2f0;--run-bg:#13233a;
  --add:#5fd39b;--del:#f08a8a;--hl:#2a2616}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;padding:28px 18px 60px;
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
header{margin-bottom:20px}
h1{font-size:26px;line-height:1.2;margin:0 0 8px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:13.5px}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.badge{display:inline-block;font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:99px;
  letter-spacing:.02em;white-space:nowrap}
.b-ok{background:var(--ok-bg);color:var(--ok)}
.b-hyp{background:var(--hyp-bg);color:var(--hyp)}
.b-run{background:var(--run-bg);color:var(--run)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 22px 18px;margin:0 0 14px}
h2{font-size:18px;margin:2px 0 10px;letter-spacing:-.01em}
h3{font-size:14.5px;margin:18px 0 6px;display:flex;align-items:center;gap:8px}
h3 .badge{font-size:12px}
p{margin:8px 0}
ul,ol{margin:6px 0;padding-left:22px}
li{margin:3px 0}
li>ul,li>ol{margin:3px 0}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);
  padding:1px 5px;border-radius:4px}
pre{background:var(--code);border-radius:8px;padding:10px 12px;overflow-x:auto;margin:10px 0}
pre code{background:none;padding:0;font-size:12.5px}
pre .add{color:var(--add)} pre .del{color:var(--del)}
.tbl{overflow-x:auto;margin:10px 0 6px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);background:var(--code);white-space:nowrap}
tr:last-child td{border-bottom:none}
tr.hl td{background:var(--hl)}
a{color:var(--run)}
footer{color:var(--muted);font-size:12px;margin-top:18px;text-align:center}
@media print{body{background:#fff;padding:0} section{break-inside:avoid;border-color:#ccc}
  .tbl{overflow:visible}}
"""


def inline(text):
    codes = []

    def keep(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
                  lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: f"<code>{html.escape(codes[int(m.group(1))], quote=False)}</code>", text)


LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def render_list(lines):
    """lines: raw lines of one list block (items + indented continuations)."""
    items = []  # (indent, ordered, text)
    for ln in lines:
        m = LIST_RE.match(ln)
        if m:
            items.append([len(m.group(1)), m.group(2)[0].isdigit(), m.group(3)])
        else:
            items[-1][2] += " " + ln.strip()
    out, stack = [], []  # stack of (indent, tag)
    for indent, ordered, text in items:
        tag = "ol" if ordered else "ul"
        while stack and indent < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or indent > stack[-1][0]:
            out.append(f"<{tag}>")
            stack.append((indent, tag))
        else:
            out.append("</li>")
        out.append(f"<li>{inline(text)}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def render_table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = []
    for r in body:
        cls = ' class="hl"' if r and r[0].startswith("**") else ""
        b.append(f"<tr{cls}>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{"".join(b)}</tbody></table></div>'


def render_code(lang, lines):
    out = []
    for ln in lines:
        e = html.escape(ln, quote=False)
        if lang == "diff" and ln.startswith("+"):
            e = f'<span class="add">{e}</span>'
        elif lang == "diff" and ln.startswith("-"):
            e = f'<span class="del">{e}</span>'
        out.append(e)
    return f'<pre><code class="language-{html.escape(lang)}">' + "\n".join(out) + "</code></pre>"


def convert(md):
    lines = md.splitlines()
    title, subtitle, body, in_section = "", "", [], False
    i, n = 0, len(lines)
    para = []

    def flush_para():
        if para:
            body.append(f"<p>{inline(' '.join(x.strip() for x in para))}</p>")
            para.clear()

    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush_para()
            lang, code = s[3:].strip(), []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            body.append(render_code(lang, code))
            i += 1
            continue
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
            i += 1
            sub = []
            while i < n and not lines[i].strip():
                i += 1
            if i < n and lines[i].startswith("*"):
                while i < n and lines[i].strip():
                    sub.append(lines[i].strip())
                    i += 1
            subtitle = " ".join(sub)
            if subtitle.startswith("*") and subtitle.endswith("*"):
                subtitle = subtitle[1:-1]
            continue
        if ln.startswith("## "):
            flush_para()
            if in_section:
                body.append("</section>")
            body.append(f"<section><h2>{inline(ln[3:].strip())}</h2>")
            in_section = True
            i += 1
            continue
        if ln.startswith("### "):
            flush_para()
            text = ln[4:].strip()
            key = next((k for k in BADGES if text.lower().startswith(k)), None)
            if key:
                head, _, rest = text.partition(" — ")
                rest_html = f" <span>{inline(rest)}</span>" if rest else ""
                body.append(f'<h3><span class="badge {BADGES[key]}">{inline(head)}</span>{rest_html}</h3>')
            else:
                body.append(f"<h3>{inline(text)}</h3>")
            i += 1
            continue
        if s == "---":
            flush_para()
            i += 1
            continue
        if s.startswith("|"):
            flush_para()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            body.append(render_table(tbl))
            continue
        if LIST_RE.match(ln):
            flush_para()
            blk = []
            while i < n:
                cur = lines[i]
                if LIST_RE.match(cur) or (cur.startswith(" ") and cur.strip() and blk):
                    blk.append(cur)
                    i += 1
                else:
                    break
            body.append(render_list(blk))
            continue
        if not s:
            flush_para()
            i += 1
            continue
        para.append(ln)
        i += 1
    flush_para()
    if in_section:
        body.append("</section>")
    return title, subtitle, "\n".join(body)


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    md = src.read_text(encoding="utf-8")
    title, subtitle, body = convert(md)
    sha = hashlib.sha256(md.encode("utf-8")).hexdigest()
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>{inline(title)}</h1>
<div class="sub">{inline(subtitle)}</div>
<div class="legend"><span class="badge b-ok">Measured</span><span class="badge b-run">Interpretation</span><span class="badge b-hyp">Caveats</span></div>
</header>
{body}
<footer>Generated from {html.escape(src.name)} (sha256 {sha[:12]}) by tools/md2html.py</footer>
</div>
</body>
</html>
"""
    dst.write_text(page, encoding="utf-8")
    print(f"wrote {dst} ({len(page)} bytes) from {src} sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
