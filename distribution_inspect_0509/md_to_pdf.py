import re
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

BASE_DIR = "/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/distribution_inspect_0509"
MD_FILE  = os.path.join(BASE_DIR, "illustrate.md")
OUT_FILE = os.path.join(BASE_DIR, "illustrate.pdf")

FONT_REG = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

def register_fonts():
    pdfmetrics.registerFont(TTFont("CJK", FONT_REG))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    # No bold variant available; map bold → same font, use size to differentiate
    registerFontFamily("CJK", normal="CJK", bold="CJK", italic="CJK", boldItalic="CJK")

def make_styles():
    base = dict(fontName="CJK", leading=18)
    return {
        "title":     ParagraphStyle("title",     **base, fontSize=17, spaceBefore=0,  spaceAfter=8,  textColor=HexColor("#111111")),
        "subtitle":  ParagraphStyle("subtitle",  **base, fontSize=10, spaceAfter=10,                 textColor=HexColor("#555555")),
        "h2":        ParagraphStyle("h2",        **base, fontSize=14, spaceBefore=14, spaceAfter=5,  textColor=HexColor("#1a1a1a")),
        "body":      ParagraphStyle("body",      **base, fontSize=10, spaceAfter=3,                  textColor=HexColor("#333333")),
        "li":        ParagraphStyle("li",        **base, fontSize=10, spaceAfter=2,   leftIndent=14, textColor=HexColor("#333333")),
        "bold_line": ParagraphStyle("bold_line", **base, fontSize=10, spaceAfter=4,                  textColor=HexColor("#333333")),
    }

def md_bold(text):
    """Convert **foo** → <b>foo</b> for reportlab Paragraph."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

import io

MAX_PX = 1400  # max width in pixels before embedding (150 dpi quality)

def make_image(src, content_w):
    path = os.path.join(BASE_DIR, src)
    if not os.path.exists(path):
        return None
    with PILImage.open(path) as img:
        iw, ih = img.size
        if iw > MAX_PX:
            scale = MAX_PX / iw
            img = img.resize((MAX_PX, int(ih * scale)), PILImage.LANCZOS)
            iw, ih = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
    tw = content_w * 0.70
    return RLImage(buf, width=tw, height=ih * tw / iw)

def build_story(lines, styles, content_w):
    story = []
    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()

        if not s:
            story.append(Spacer(1, 4))
            continue

        if s == "---":
            story.append(HRFlowable(width="100%", thickness=0.4,
                                    color=HexColor("#cccccc"), spaceBefore=4, spaceAfter=4))
            continue

        if s.startswith("# ") and not s.startswith("## "):
            story.append(Paragraph(s[2:].strip(), styles["title"]))
            continue

        if s.startswith("## "):
            story.append(Paragraph(s[3:].strip(), styles["h2"]))
            continue

        if s.startswith("<img "):
            m = re.search(r'src="([^"]+)"', s)
            if m:
                img = make_image(m.group(1), content_w)
                if img:
                    story.append(Spacer(1, 4))
                    story.append(img)
                    story.append(Spacer(1, 6))
            continue

        # numbered list item
        m = re.match(r'^(\d+)\.\s+(.*)', s)
        if m:
            story.append(Paragraph(f"{m.group(1)}. {md_bold(m.group(2))}", styles["li"]))
            continue

        # bold-only line like **分难度审查**：
        if s.startswith("**") and "**" in s[2:]:
            story.append(Paragraph(md_bold(s), styles["bold_line"]))
            continue

        story.append(Paragraph(md_bold(s), styles["body"]))

    return story

def main():
    register_fonts()
    st = make_styles()

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    content_w = PAGE_W - 2 * MARGIN

    doc = SimpleDocTemplate(
        OUT_FILE, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    with open(MD_FILE, encoding="utf-8") as f:
        lines = f.readlines()

    story = build_story(lines, st, content_w)
    doc.build(story)
    print(f"Done → {OUT_FILE}")

if __name__ == "__main__":
    main()
