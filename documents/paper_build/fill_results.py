# -*- coding: utf-8 -*-
"""Fill the paper's result placeholders from the local GPU run (2026-11-23)."""
import sys
from pathlib import Path

p = Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
R = []


def rep(a, b):
    R.append((a, b))


# Abstract
rep('''"0 to 2 times. The second is 2,521 AI-generated campus scenes, produced with SDXL-Turbo from a fixed prompt plan, "
    "filtered with CLIP and matched to TrashNet's class counts. On these, MobileNetV2 reached [[X.XX%]] macro F1, and "
    "fine-tuning on TrashNet plus generated images gave [[X.XX%]]. Strong benchmark scores did not carry over to real "
    "photographs. The results motivate a larger real-photo collection before any campus use."''',
'''"0 to 2 times. The second is 2,517 AI-generated campus scenes, produced with SDXL-Turbo from a fixed prompt plan, "
    "filtered with CLIP and matched to TrashNet's class counts. On these, MobileNetV2 fell to 59.73% macro F1. "
    "Fine-tuning on TrashNet plus generated images raised macro F1 on the generated test split to 87.66%, kept "
    "TrashNet test macro F1 at 82.45%, and gave 5 of 9 correct real photos. Strong benchmark scores did not carry over "
    "to everyday scenes. The results motivate a larger real-photo collection before any campus use."''')
# Counts in intro and methodology
rep('''"was not feasible in the available time, and with the instructor's approval, we generated 2,521 campus-like "''',
    '''"was not feasible in the available time, and with the instructor's approval, we generated 2,517 campus-like "''')
rep('''"that disclosure in its metadata. Class targets copy TrashNet's usable counts exactly (Table 1), so differences "
      "between the datasets reflect the images rather than sample size or class balance."),''',
'''"that disclosure in its metadata. Class targets copy TrashNet's usable counts (Table 1), so differences between "
      "the datasets reflect the images rather than sample size or class balance. Five classes met their targets "
      "exactly. Plastic reached 476 of 480, which the protocol accepts (at least 95% of the target), so 2,517 images "
      "were kept. No settings were changed after generation."),''')
rep('''Generation.** SDXL-Turbo [13] (revision 71153311) produced 1.8 candidates per target image, 4,541 in total, "
      "with 4 inference steps, guidance 0 and one fixed seed per candidate.''',
'''Generation.** SDXL-Turbo [13] (revision 71153311) produced 1.8 candidates per target image, 4,541 in total, "
      "with 4 inference steps, guidance 0 and one fixed seed per candidate, on one NVIDIA RTX 2080 GPU (about "
      "1.8 hours).''')
rep('''"across item types until each class reached its target. Of 4,541 candidates, [[N]] failed the CLIP rule, [[N]] "
      "were near-duplicates and [[N]] were surplus."),''',
'''"across item types until each class reached its target. Of 4,541 candidates, 1,428 failed the CLIP rule (1,298 "
      "with the wrong top class, 105 ambiguous, 25 low-confidence), 1 was a near-duplicate and 595 were surplus."),''')
rep('''"TrashNet, giving [[1,765 / 378 / 378]] training, validation and test images. Groups never cross partitions."),''',
'''"TrashNet, giving 1,765 training, 361 validation and 391 test images from 640 groups. Groups never cross "
      "partitions."),''')
rep('"real photographs, on all 2,521 generated images and on the generated test split.',
    '"real photographs, on all 2,517 generated images and on the generated test split.')
# Real photos paragraph
rep('''"models did no better: weighted CNN 2/9, small CNN 1/9, SVM 1/9 and Random Forest 0/9 [[confirm the four "
      "non-MobileNet counts in the pinned Colab environment]]. With nine photos and three classes, these are case "
      "observations, not an accuracy estimate."),''',
'''"models did no better: weighted CNN 2/9, small CNN 1/9, SVM 1/9 and Random Forest 0/9. With nine photos and "
      "three classes, these are case observations, not an accuracy estimate."),''')
# RQ2
rep('''("p", "[[Fill from reports/primary_set/dataset_comparison.json and dataset_statistics.csv. Expected points to "
      "check: white-background fraction (TrashNet is mostly white posterboard), saturation and edge density "
      "(cluttered scenes), and the domain-separability accuracy.]]"),''',
'''("p", "The two datasets have the same size and class mix (trash is 5.4% of each) but look very different (Table 5, "
      "Figure 2). The generated scenes are darker (mean brightness 0.44 vs 0.64), more than twice as saturated "
      "(0.39 vs 0.18) and busier (edge density 0.21 vs 0.13). Near-white pixels cover 8.5% of a TrashNet image on "
      "average but 1.8% of a generated one. A logistic-regression classifier on ImageNet MobileNetV2 features told "
      "the two datasets apart with 98.97% cross-validated accuracy, where 50% would mean no difference. The primary "
      "data therefore represent a clearly different domain from the benchmark, as intended."),''')
rep('''("pending_figure", "Figure 2. Random examples per class: TrashNet (left) and generated primary images (right). "
                   "[[Insert reports/primary_set/sample_grid.png]]"),''',
'''("figure", "fig2"),''')
# RQ3 on generated images
rep('''("p", "[[Fill from reports/primary_set/domain_shift.csv: macro F1 of each frozen model on the generated images, the change "
      "from its TrashNet test score, trash recall, and the hardest settings from domain_shift.json "
      "(mobilenet_accuracy_by_setting).]]"),''',
'''("p", "Every frozen TrashNet model lost accuracy on the generated images (Table 6). The HOG pipelines collapsed to "
      "10–11% macro F1, below chance for six classes, and mostly predicted cardboard. The small CNNs fell to 20–22%. "
      "MobileNetV2 kept the most, 59.73% macro F1 (64.76% accuracy), a drop of 21.5 points. Its recall stayed high "
      "for cardboard (91%) and glass (86%) but fell to 40% for plastic and 20% for trash, the same classes that "
      "failed on the real photographs. Accuracy was highest for the plain-white-background control (72.6%) and lowest "
      "beside outdoor bins (58.4%) and on fallen leaves (60.5%), so background clutter accounts for part of the drop."),''')
# RQ4
rep('''("p", "[[Fill from reports/primary_set/finetune_comparison.csv: does adding generated training images raise generated-test "
      "test F1, and does TrashNet test F1 stay level or fall?]]"),''',
'''("p", "Adding 1,765 generated training images helped on all three test sets (Table 7). The fine-tuned model was "
      "selected on combined validation macro F1 (84.47%, epoch 12) before any test evaluation. On the generated test "
      "split, macro F1 rose from 58.50% to 87.66% and trash recall from 22.22% to 77.78%. On the TrashNet test split, "
      "macro F1 did not fall: it moved from 81.22% to 82.45%, with trash recall rising from 55.56% to 66.67% (10 to 12 "
      "of 18). On the real photographs, the fine-tuned model was correct on 5 of 9 instead of 3 (Table 4). It newly "
      "recognised the paper shopping bag and the soft-drink bottle and lost none of the earlier correct answers. "
      "Plastic film, bags and packaging were still misread as paper or cardboard. With 9 photos the gain is "
      "suggestive, not statistically reliable."),''')
# Discussion and conclusion
rep('''"cardboard, and a clear bottle as glass. [[Interpret RQ2–RQ4 from the generated-image results: whether the "
      "generated scenes reproduce this gap, and whether training on them improves the real-photo outcomes.]]"),''',
'''"cardboard, and a clear bottle as glass. The generated scenes reproduce the same pattern at scale: plastic and "
      "trash were the weakest classes there too, and clutter lowered accuracy. Training on generated scenes closed most "
      "of the gap on generated images without costing TrashNet accuracy, and it moved real-photo results in the right "
      "direction. This suggests that varied backgrounds, not only more images, are what the benchmark lacks. Soft "
      "plastic film remained unsolved, possibly because the generated plastic class, like TrashNet's, is dominated by "
      "rigid bottles and cups."),''')
rep('"trash poorly and was correct on only 3 of 9 real photographs taken by the group. A matched set of 2,521 generated "',
    '"trash poorly and was correct on only 3 of 9 real photographs taken by the group. A matched set of 2,517 generated "')
rep('''"campus-like images makes this gap measurable at scale. [[Summarise the generated-image and fine-tuning outcome "
      "in one sentence.]] A validation set''',
'''"campus-like images reproduced this gap (59.73% macro F1), and adding them to training raised generated-test macro "
      "F1 to 87.66% while keeping TrashNet performance (82.45%) and improving the real-photo count to 5 of 9. A "
      "validation set''')
# Tables
rep('''["Paper", "594", "2", "594"], ["Plastic", "480", "6", "480"], ["Trash", "137", "0", "137"],
            ["**Total**", "**2,521**", "**9 (+2)**", "**2,521**"]],''',
'''["Paper", "594", "2", "594"], ["Plastic", "480", "6", "476"], ["Trash", "137", "0", "137"],
            ["**Total**", "**2,521**", "**9 (+2)**", "**2,517**"]],''')
rep('''[["Brightness", P, P], ["Contrast", P, P], ["Saturation", P, P], ["White-background fraction", P, P],
            ["Colourfulness", P, P], ["Edge density", P, P], ["Domain separability (acc.)", P, ""]]),''',
'''[["Brightness", "0.640", "0.436"], ["Contrast", "0.180", "0.166"], ["Saturation", "0.178", "0.389"],
            ["White-background fraction", "0.085", "0.018"], ["Colourfulness", "0.100", "0.149"],
            ["Edge density", "0.130", "0.207"]],
           "A domain classifier told the datasets apart with 98.97% accuracy (50% = indistinguishable)."),''')
rep('''[["HOG + SVM", "63.52", P, P, P], ["HOG + Random Forest", "60.22", P, P, P],
            ["Small CNN", "57.68", P, P, P], ["MobileNetV2", "81.22", P, P, P],
            ["Small CNN, weighted", "58.84", P, P, P]]),''',
'''[["HOG + SVM", "63.52", "11.37", "−52.15", "0.73"], ["HOG + Random Forest", "60.22", "10.02", "−50.20", "0.00"],
            ["Small CNN", "57.68", "22.31", "−35.37", "4.38"],
            ["**MobileNetV2**", "**81.22**", "**59.73**", "**−21.49**", "**20.44**"],
            ["Small CNN, weighted", "58.84", "20.04", "−38.79", "8.03"]],
           "All 2,517 generated images; models unchanged. Trash recall in %."),''')
rep('''[["TrashNet", "TrashNet test", "84.57", "81.22", "55.56"], ["TrashNet", "Generated test", P, P, P], ["TrashNet", "Real photos (9)", "33.33", "–", "–"],
            ["TrashNet + generated", "TrashNet test", P, P, P], ["TrashNet + generated", "Generated test", P, P, P], ["TrashNet + generated", "Real photos (9)", P, "–", "–"]]),''',
'''[["TrashNet", "TrashNet test", "84.57", "81.22", "55.56"], ["TrashNet", "Generated test", "63.43", "58.50", "22.22"],
            ["TrashNet", "Real photos (9)", "33.33", "–", "–"],
            ["TrashNet + generated", "TrashNet test", "**84.84**", "**82.45**", "**66.67**"],
            ["TrashNet + generated", "Generated test", "**90.54**", "**87.66**", "**77.78**"],
            ["TrashNet + generated", "Real photos (9)", "**55.56**", "–", "–"]],
           "Test splits: TrashNet n = 376, generated n = 391. The real photos cover three classes, so macro F1 is "
           "not reported."),''')
rep('''TABLES["t7"] = ("Table 4. MobileNetV2 on the group's real photographs",
           ["Photo", "Item (actual)", "Predicted", "Score"],
           [700, 2000, 1100, 900],
           [["R1", "Courier pouch (plastic)", "Paper", "0.68"], ["R2", "Food can (metal)", "**Metal**", "0.83"],
            ["R3", "Pump nozzle (plastic)", "Cardboard", "0.45"], ["R4", "Shopping bag (paper)", "Cardboard", "0.57"],
            ["R7", "Crumpled bag (paper)", "**Paper**", "0.64"], ["R8", "Food tub (plastic)", "**Plastic**", "0.73"],
            ["R9", "Garbage bag (plastic)", "Cardboard", "0.73"], ["R10", "Drink bottle (plastic)", "Glass", "0.72"],
            ["R11", "Air cushions (plastic)", "Paper", "0.53"], ["**Total**", "", "**3 / 9**", ""]],
           "Bold = correct. Score = top softmax output (uncalibrated).")''',
'''TABLES["t7"] = ("Table 4. MobileNetV2 predictions on the group's real photographs",
           ["Photo", "Item (actual)", "TrashNet only", "+ generated"],
           [550, 1750, 1250, 1250],
           [["R1", "Courier pouch (plastic)", "Paper 0.68", "Paper 0.81"],
            ["R2", "Food can (metal)", "**Metal 0.83**", "**Metal 0.91**"],
            ["R3", "Pump nozzle (plastic)", "Cardboard 0.45", "Cardboard 0.42"],
            ["R4", "Shopping bag (paper)", "Cardboard 0.57", "**Paper 0.71**"],
            ["R7", "Crumpled bag (paper)", "**Paper 0.64**", "**Paper 0.57**"],
            ["R8", "Food tub (plastic)", "**Plastic 0.73**", "**Plastic 0.96**"],
            ["R9", "Garbage bag (plastic)", "Cardboard 0.73", "Cardboard 0.61"],
            ["R10", "Drink bottle (plastic)", "Glass 0.72", "**Plastic 0.70**"],
            ["R11", "Air cushions (plastic)", "Paper 0.53", "Paper 0.58"],
            ["**Correct**", "", "**3 / 9**", "**5 / 9**"]],
           "Bold = correct. Number = top softmax output (uncalibrated). “+ generated” = fine-tuned on TrashNet "
           "plus generated images.")''')
rep('''FIGURES = {"fig1": ("mobilenet_confusion.png",
                    "Figure 1. MobileNetV2 confusion matrix on the TrashNet test split (rows: actual; columns: predicted).")}''',
'''FIGURES = {"fig1": ("mobilenet_confusion.png",
                    "Figure 1. MobileNetV2 confusion matrix on the TrashNet test split (rows: actual; columns: predicted)."),
           "fig2": ("primary_set/sample_grid.png",
                    "Figure 2. Random examples per class: TrashNet (left) and generated primary images (right).")}''')

for a, b in R:
    assert a in t, a[:100]
    t = t.replace(a, b, 1)
p.write_text(t, encoding="utf-8")
print("ok; remaining placeholders:")
for line in t.splitlines():
    if "[[" in line:
        print("  ", line.strip()[:110])
