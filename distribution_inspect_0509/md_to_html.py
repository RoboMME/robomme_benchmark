"""Convert illustrate.md to a self-contained HTML file for Chrome headless PDF printing.
Images are downsampled to MAX_PX wide before base64-embedding to keep file size reasonable.
"""
import re, os, base64, io
from PIL import Image as PILImage

BASE_DIR = "/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/distribution_inspect_0509"
MD_FILE  = os.path.join(BASE_DIR, "illustrate.md")
OUT_HTML = os.path.join(BASE_DIR, "illustrate.html")


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei",
                 "WenQuanYi Micro Hei", "Droid Sans Fallback", sans-serif;
    font-size: 13px;
    line-height: 1.7;
    color: #333;
    padding: 24mm 20mm 20mm 20mm;
    max-width: 210mm;
    margin: 0 auto;
}
h1 { font-size: 20px; font-weight: 700; margin-bottom: 6px; color: #111; }
h2 { font-size: 15px; font-weight: 700; margin-top: 0; margin-bottom: 6px;
     color: #1a1a1a; border-bottom: 1px solid #ddd; padding-bottom: 3px;
     page-break-before: always; break-before: page; }
h2.first { page-break-before: avoid; break-before: avoid; }
hr { border: none; border-top: 1px solid #ddd; margin: 8px 0; }
p  { margin-bottom: 4px; }
ol { margin: 4px 0 6px 18px; }
li { margin-bottom: 2px; }
img { display: block; max-width: 70%; height: auto; margin: 8px 0; }
strong { font-weight: 700; }
@page { size: A4; margin: 0; }
@media print { body { padding: 18mm 18mm 16mm 18mm; } }
"""

def img_to_data_uri(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline_md(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text

def md_to_html(lines):
    parts = []
    h2_count = 0
    i = 0
    while i < len(lines):
        s = lines[i].rstrip("\n").strip()

        if not s:
            i += 1
            continue

        if s == "---":
            parts.append("<hr>")
            i += 1
            continue

        if s.startswith("# ") and not s.startswith("## "):
            parts.append(f"<h1>{escape(s[2:].strip())}</h1>")
            i += 1
            continue

        if s.startswith("## "):
            cls = " class=\"first\"" if h2_count == 0 else ""
            h2_count += 1
            parts.append(f"<h2{cls}>{escape(s[3:].strip())}</h2>")
            i += 1
            continue

        if s.startswith("<img "):
            m = re.search(r'src="([^"]+)"', s)
            if m:
                path = os.path.join(BASE_DIR, m.group(1))
                if os.path.exists(path):
                    uri = img_to_data_uri(path)
                    alt = (re.search(r'alt="([^"]*)"', s) or type('', (), {'group': lambda self, n: ''})()).group(1)
                    parts.append(f'<img src="{uri}" alt="{escape(alt)}">')
            i += 1
            continue

        # Numbered list — collect consecutive items
        if re.match(r'^\d+\.', s):
            items = []
            while i < len(lines):
                s2 = lines[i].rstrip("\n").strip()
                m2 = re.match(r'^(\d+)\.\s+(.*)', s2)
                if m2:
                    items.append(inline_md(m2.group(2)))
                    i += 1
                else:
                    break
            parts.append("<ol>" + "".join(f"<li>{t}</li>" for t in items) + "</ol>")
            continue

        parts.append(f"<p>{inline_md(s)}</p>")
        i += 1

    return "\n".join(parts)

def main():
    with open(MD_FILE, encoding="utf-8") as f:
        lines = f.readlines()

    body = md_to_html(lines)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>5/9 robomme env 分布审查</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML → {OUT_HTML}")

if __name__ == "__main__":
    main()
