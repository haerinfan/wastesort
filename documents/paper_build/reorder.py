# -*- coding: utf-8 -*-
"""Put Results in RQ order (RQ1, RQ2, RQ3, RQ4) and renumber Tables 4 and 5 to match."""
import sys
from pathlib import Path

p = Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
real_h = '("h2", "The Group\'s Real Photographs (RQ3)"),'
rq2_h = '("h2", "Primary versus Secondary Dataset (RQ2)"),'
gen_h = '("h2", "Domain Shift on Generated Images (RQ3)"),'
a, b, c = t.index(real_h), t.index(rq2_h), t.index(gen_h)
assert a < b < c
real_block, rq2_block = t[a:b], t[b:c]
# Swap table references before moving: real photos become Table 5, image statistics Table 4.
assert real_block.count("(Table 4)") == 1
real_block = real_block.replace("(Table 4)", "(Table 5)")
assert rq2_block.count("(Table 5, ") == 1
rq2_block = rq2_block.replace("(Table 5, ", "(Table 4, ")
real_block = real_block.replace(real_h, '("h2", "Domain Shift: The Group\'s Real Photographs (RQ3)"),')
t = t[:a] + rq2_block + real_block + t[c:]
t = t.replace(gen_h, '("h2", "Domain Shift: Generated Images (RQ3)"),', 1)
# RQ4 paragraph refers to the real-photo table.
old = "of 18). On the real photographs, the fine-tuned model was correct on 5 of 9 instead of 3 (Table 4)."
assert old in t
t = t.replace(old, old.replace("(Table 4)", "(Table 5)"))
# Table captions.
for x, y in [('"Table 5. Mean image statistics', '"Table 4. Mean image statistics'),
             ('"Table 4. MobileNetV2 predictions on the group\'s real photographs"',
              '"Table 5. MobileNetV2 predictions on the group\'s real photographs"')]:
    assert x in t, x
    t = t.replace(x, y, 1)
p.write_text(t, encoding="utf-8", newline="\n")
print("ok")
