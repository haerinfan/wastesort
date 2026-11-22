# -*- coding: utf-8 -*-
"""Build the WasteSort ACM paper by refilling the user's ACM template body. Usage: build.py TEMPLATE_DIR REPO OUT"""
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import content as C

template, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
work = Path(__file__).parent / "acm_build"
if work.exists():
    shutil.rmtree(work)
shutil.copytree(template, work)
doc_path = work / "word/document.xml"
xml = doc_path.read_text(encoding="utf-8")
head = xml[: xml.index("<w:body>") + len("<w:body>")]
sects = re.findall(r"<w:sectPr.*?</w:sectPr>", xml, flags=re.S)
sect_title, sect_authors, sect_body, sect_final = sects

COLUMN_TWIPS = 4800


def runs(text, base=""):
    """*italic*, **bold**, [[placeholder]] -> runs."""
    out = []
    for token in re.split(r"(\*\*.+?\*\*|\*.+?\*|\[\[.+?\]\])", text):
        if not token:
            continue
        props = base
        if token.startswith("**"):
            token, props = token[2:-2], props + "<w:b/><w:bCs/>"
        elif token.startswith("[["):
            token, props = "[" + token[2:-2] + "]", props + '<w:highlight w:val="yellow"/>'
        elif token.startswith("*") and token.endswith("*") and len(token) > 1:
            token, props = token[1:-1], props + "<w:i/><w:iCs/>"
        rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(token)}</w:t></w:r>')
    return "".join(out)


def para(text, style=None, ppr="", base=""):
    s = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{s}{ppr}</w:pPr>{runs(text, base)}</w:p>"


def body(text):
    return para(text, "BodyTextIndent", '<w:spacing w:after="120"/><w:ind w:firstLine="0"/><w:jc w:val="both"/>')


def sect_para(sect):
    return f"<w:p><w:pPr>{sect}</w:pPr></w:p>"


def table(key):
    spec = C.TABLES[key]
    caption, header, widths, rows = spec[:4]
    note = spec[4] if len(spec) > 4 else None
    assert sum(widths) <= COLUMN_TWIPS, key
    parts = [para(caption, "Caption", "<w:keepNext/><w:spacing w:before=\"120\" w:after=\"60\"/>")]
    borders = "".join(f'<w:{b} w:val="single" w:sz="2" w:space="0" w:color="000000"/>'
                      for b in ["top", "left", "bottom", "right", "insideH", "insideV"])
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    trs = []
    for r, row in enumerate([header] + rows):
        cells = []
        for c, (value, width) in enumerate(zip(row, widths)):
            text = f"**{value}**" if r == 0 and value else value
            align = "left" if c == 0 else "center"
            keep = "<w:keepNext/>" if r < len(rows) else ""  # keep each table in one column
            cell_p = para(text, "BodyTextIndent", f'{keep}<w:ind w:firstLine="0"/><w:jc w:val="{align}"/>',
                          base='<w:sz w:val="16"/><w:szCs w:val="16"/>')
            cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:vAlign w:val="center"/>'
                         f'</w:tcPr>{cell_p}</w:tc>')
        header_flag = "<w:tblHeader/>" if r == 0 else ""
        trs.append(f"<w:tr><w:trPr><w:cantSplit/>{header_flag}</w:trPr>{''.join(cells)}</w:tr>")
    parts.append(f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="center"/>'
                 f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
                 f'<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
                 f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
    after = note or ""
    parts.append(para(after, "BodyTextIndent", '<w:spacing w:before="40" w:after="120"/><w:ind w:firstLine="0"/>',
                      base='<w:sz w:val="16"/><w:szCs w:val="16"/>'))
    return "".join(parts)


images = []


def figure(key):
    name, caption = C.FIGURES[key]
    source = repo / "reports" / name
    target = work / "word/media" / f"paper_{name}"
    shutil.copyfile(source, target)
    width_px, height_px = Image.open(source).size
    cx = int(3.15 * 914400)
    cy = int(cx * height_px / width_px)
    rid = f"rIdPaper{len(images) + 1}"
    images.append((rid, f"media/paper_{name}"))
    n = 100 + len(images)
    drawing = (
        f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{n}" name="Figure {n}" descr="{escape(caption)}"/><wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        '</wp:inline></w:drawing></w:r>')
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr>'
            f'{drawing}</w:p>' + para(caption, "Caption", '<w:spacing w:after="160"/>'))


def pending_figure(caption):
    box = ('<w:pBdr><w:top w:val="dashed" w:sz="4" w:space="4" w:color="999999"/>'
           '<w:bottom w:val="dashed" w:sz="4" w:space="4" w:color="999999"/></w:pBdr>')
    return (para("[[Figure 2 goes here after the Colab run]]", None,
                 f'<w:keepNext/>{box}<w:spacing w:before="240" w:after="240"/><w:jc w:val="center"/>')
            + para(caption, "Caption", '<w:spacing w:after="160"/>'))


parts = [para(C.TITLE, "Paper-Title", '<w:spacing w:after="60"/>'), sect_para(sect_title)]
for i, (name, email) in enumerate(C.AUTHORS):
    brk = '<w:r><w:br w:type="column"/></w:r>' if i else ""
    parts.append(f'<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:after="0"/></w:pPr>{brk}{runs(name)}</w:p>')
    lines = "".join(f'<w:r>{"<w:br/>" if k else ""}<w:t xml:space="preserve">{escape(line)}</w:t></w:r>'
                    for k, line in enumerate(C.AFFILIATION))
    parts.append(f'<w:p><w:pPr><w:pStyle w:val="Affiliations"/></w:pPr>{lines}</w:p>')
    parts.append(para(email, "E-Mail"))
parts.append(para("", "E-Mail"))
parts.append(f'<w:p><w:pPr><w:jc w:val="center"/>{sect_authors}</w:pPr></w:p>')
heading = '<w:spacing w:before="120" w:after="0"/>'
parts.append(para("ABSTRACT", None, '<w:spacing w:after="0"/>', base='<w:b/><w:sz w:val="24"/>'))
parts.append(para(C.ABSTRACT, "Abstract"))
parts.append(para("CCS Concepts", None, heading, base='<w:b/><w:sz w:val="24"/>'))
ccs = "   • ".join(f"**{a}**➝**{b}**" for a, b in C.CCS)
parts.append(para("• " + ccs, None, '<w:spacing w:after="120"/><w:jc w:val="left"/>'))
parts.append(para("Keywords", None, heading, base='<w:b/><w:sz w:val="24"/>'))
parts.append(para(C.KEYWORDS, None, '<w:spacing w:after="120"/>'))
for kind, payload in C.BODY:
    if kind == "h1":
        parts.append(para(payload, "Heading1", '<w:spacing w:before="120"/>'))
    elif kind == "h2":
        parts.append(para(payload, "Heading2", '<w:spacing w:before="120"/>'))
    elif kind == "p":
        parts.append(body(payload))
    elif kind == "bullets":
        for item in payload:
            parts.append(para("• " + item, "Bullet", '<w:spacing w:after="60"/><w:ind w:left="216" w:hanging="216"/><w:jc w:val="both"/>'))
        parts.append(para("", None, '<w:spacing w:after="60"/>'))
    elif kind == "table":
        parts.append(table(payload))
    elif kind == "figure":
        parts.append(figure(payload))
    elif kind == "pending_figure":
        parts.append(pending_figure(payload))
parts.append(para("REFERENCES", "Heading1", '<w:spacing w:before="120"/>'))
for ref in C.REFERENCES:
    parts.append(para(ref, "References"))
parts.append(f'<w:p><w:pPr><w:pStyle w:val="References"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>'
             f'{sect_body}</w:pPr></w:p>')
doc_path.write_text(head + "".join(parts) + sect_final + "</w:body></w:document>", encoding="utf-8")

rels_path = work / "word/_rels/document.xml.rels"
rels = rels_path.read_text(encoding="utf-8")
new = "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
              f'Target="{target}"/>' for rid, target in images)
rels_path.write_text(rels.replace("</Relationships>", new + "</Relationships>"), encoding="utf-8")

core = work / "docProps/core.xml"
if core.exists():
    text = core.read_text(encoding="utf-8")
    text = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(C.TITLE)}</dc:title>", text, flags=re.S)
    text = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator>Goal Diggers</dc:creator>", text, flags=re.S)
    core.write_text(text, encoding="utf-8")

out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    out.unlink()
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(work.rglob("*")):
        if path.is_file():
            name = path.relative_to(work).as_posix()
            z.write(path, name, compress_type=zipfile.ZIP_STORED if name == "[Content_Types].xml" else zipfile.ZIP_DEFLATED)
print("wrote", out)
