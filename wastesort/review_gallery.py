"""Create a portable visual worksheet for candidate pairs; do not change labels or splits."""
import base64
import html
import io

import pandas as pd
from PIL import Image, ImageOps

from .common import ROOT


def thumbnail(relative):
    with Image.open(ROOT / relative) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((512, 384))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    pairs = pd.read_csv(ROOT / "reports/near_duplicate_candidates.csv")
    manifest = pd.read_csv(ROOT / "data/splits.csv").set_index("path")
    cards = []
    for i, row in enumerate(pairs.itertuples(), 1):
        split = manifest.loc[row.left, "split"]
        assert split == manifest.loc[row.right, "split"]
        figures = "".join(
            f'<figure><img src="{thumbnail(relative)}" alt="{html.escape(relative)}">'
            f'<figcaption>{html.escape(relative)}</figcaption></figure>'
            for relative in [row.left, row.right])
        cards.append(f'<article id="pair-{i}"><h2>Pair {i:02d}</h2>'
                     f'<p>pHash distance: {row.hamming_distance} / 64 · Same label: {row.same_label} · Partition: {split}</p>'
                     f'<div class="pair">{figures}</div>'
                     '<p class="review">Review: compare object details, markings, shape and scene. Record your decision '
                     'in manual_review_worksheet.csv. Visual similarity alone does not establish the same physical object.</p></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WasteSort candidate-pair review</title><style>
body{font:17px/1.5 Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;background:#f7f8f2;color:#183c2d}
article{border-top:1px solid #ccd5cb;padding:20px 0;break-inside:avoid}h1{font-size:34px}.pair{display:flex;gap:20px}
figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto;display:block}figcaption{overflow-wrap:anywhere;font-size:14px}
.review{font-size:14px;color:#54675a}@media(max-width:600px){.pair{display:block}figure{margin-bottom:16px}}
@media print{body{margin:0;max-width:none}article{page-break-inside:avoid}}
</style><h1>WasteSort candidate-pair review</h1>
<p>Goal Diggers · 45 automated similarity candidates. This gallery is not a completed human audit.</p>
<p>All candidate groups already stay within one partition. The published test set has been evaluated: do not change
its labels or membership and report the revised score as if it used an untouched test set. Record review findings
for a separately versioned study. Keep uncertain cases marked uncertain.</p>
<p>Images: TrashNet by Gary Thung and Mindy Yang. See THIRD_PARTY_NOTICES.md for the source MIT notice.
The gallery embeds thumbnails and works offline. It does not save decisions or transmit data.</p>
'''
    destination = ROOT / "reports/candidate_review.html"
    destination.write_bytes((page + "".join(cards) + "</html>").encode("utf-8"))
    print(f"Created 45-pair review gallery: {destination.name}")


if __name__ == "__main__":
    main()
