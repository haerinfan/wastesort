# -*- coding: utf-8 -*-
"""Restructure the paper into the instructor's IMRAD sections (Week 16 instructions)."""
import sys
from pathlib import Path

p = Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
R = []


def rep(a, b):
    R.append((a, b))


# 1. Introduction: background, objectives, significance, related work as subsections.
rep('''("h1", "INTRODUCTION"),
("p", "Sorting waste correctly''', '''("h1", "INTRODUCTION"),
("h2", "Background"),
("p", "Sorting waste correctly''')
rep('''("p", "We address four research questions:"),''',
'''("h2", "Objectives"),
("p", "The general objective is to build and evaluate supervised image classifiers that identify the material of a "
      "single waste item, and to test whether benchmark performance holds outside the benchmark. The specific "
      "objectives answer four research questions:"),''')
rep('''("p", "Our contributions are a reproducible benchmark with documented duplicate handling and validation-only model "
      "selection, a real-photo test of the benchmark models, a fully recorded procedure for generating and filtering "
      "campus-like images, and a matched comparison that measures the gap between studio-style photographs and "
      "everyday scenes."),

("h1", "RELATED WORK"),''',
'''("h2", "Significance"),
("p", "Misplaced recyclables lower the value of collected material, and sorting guidance on campus is usually a "
      "static poster. A camera-based helper could give feedback at the bin, but only if it works on the photographs "
      "students actually take. This study shows how far a benchmark score can be trusted for that purpose and which "
      "materials fail first. Its contributions are a reproducible benchmark with documented duplicate handling and "
      "validation-only model selection, a real-photo test of the benchmark models, a fully recorded procedure for "
      "generating and filtering campus-like images, and a matched comparison that measures the gap between "
      "studio-style photographs and everyday scenes. All code, data records and results are public so other student "
      "groups can repeat or extend the work."),
("h2", "Related Work"),''')
# 2. Methods heading
rep('("h1", "METHODOLOGY"),', '("h1", "METHODS"),')
# 3. Results: training visualisation, then renumber figures.
rep('''"+17.25 to +30.21 points. The HOG pipelines and the small CNN trailed by 18 to 24 points of macro F1."),
("table", "t2"),''',
'''"+17.25 to +30.21 points. The HOG pipelines and the small CNN trailed by 18 to 24 points of macro F1. Training was "
      "stable (Figure 1): loss fell steadily, validation macro F1 peaked at epoch 8, and early stopping ended training "
      "at epoch 13."),
("table", "t2"),
("figure", "fig0"),''')
rep('"Per-class results (Table 3, Figure 1) show', '"Per-class results (Table 3, Figure 2) show')
rep('look very different (Table 5, "\n      "Figure 2).', 'look very different (Table 5, "\n      "Figure 3).')
# 4. Discussion: interpretation, limitations, recommended improvements.
rep('''("h1", "DISCUSSION"),
("p", "On TrashNet,''', '''("h1", "DISCUSSION"),
("h2", "Interpretation of Findings"),
("p", "On TrashNet,''')
rep('''("h1", "LIMITATIONS AND ETHICS"),''', '''("h2", "Limitations and Ethics"),''')
rep('''"cleanliness or safety, and the prototype must not replace posted disposal rules."),''',
'''"cleanliness or safety, and the prototype must not replace posted disposal rules."),
("h2", "Recommended Improvements"),
("p", "The findings point to five concrete improvements, in order of expected benefit:"),
("bullets", [
    "**Collect real campus photographs.** At least 50 photos per class, taken by different people at real bins, "
    "split by item so no object appears in two partitions, used first for validation and then as a test set.",
    "**Cover soft plastic.** Add film, bags, pouches and packaging to the prompt plan, since these failed in every "
    "test, and have people review the generated contact sheet to remove wrong-material images.",
    "**Strengthen the trash class.** It is the smallest and most varied class; add examples and apply class "
    "weighting to MobileNetV2, which helped the small CNN.",
    "**Adapt more of the network.** Unfreeze the last MobileNetV2 blocks with a low learning rate, and compare a "
    "larger pretrained backbone under the same protocol.",
    "**Report uncertainty in the app.** Calibrate scores (for example by temperature scaling [5]) and show "
    "\\u201cnot sure\\u201d instead of a label when the top score is low. Repeat training over several seeds and "
    "splits to measure variance."]),''')
# 5. Conclusion and future work.
rep('''("h1", "CONCLUSION"),''', '''("h1", "CONCLUSION AND FUTURE WORK"),''')
rep('''"F1 to 87.66% while keeping TrashNet performance (82.45%) and improving the real-photo count to 5 of 9. A "
      "validation set of real campus photographs, with verified disposal rules, remains the necessary "
      "next step."),''',
'''"F1 to 87.66% while keeping TrashNet performance (82.45%) and improving the real-photo count to 5 of 9. These are "
      "initial results from one seed and one split, but they meet the study's objectives: the benchmark is "
      "measured, the gap to everyday scenes is quantified, and one way to narrow it is demonstrated."),
("p", "Future work will collect a labelled set of real campus photographs, extend generation to soft plastics, "
      "fine-tune more of the network, and test the prototype with students at campus bins using verified local "
      "disposal rules. Only a model that performs well on real campus photographs should be offered as sorting "
      "guidance."),''')
# Figures
rep('''FIGURES = {"fig1": ("mobilenet_confusion.png",
                    "Figure 1. MobileNetV2''', '''FIGURES = {"fig0": ("mobilenet_learning.png",
                    "Figure 1. MobileNetV2 training loss and validation macro F1 per epoch on TrashNet."),
           "fig1": ("mobilenet_confusion.png",
                    "Figure 2. MobileNetV2''')
rep('''"Figure 2. Random examples per class: TrashNet''', '''"Figure 3. Random examples per class: TrashNet''')

for a, b in R:
    assert a in t, a[:100]
    t = t.replace(a, b, 1)
t = t.replace("\\u201c", "“").replace("\\u201d", "”")
p.write_text(t, encoding="utf-8", newline="\n")
print("ok")
