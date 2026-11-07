# Evaluate new photos separately

The original TrashNet benchmark is finished. This workflow evaluates the saved model on newly labeled photographs without changing that benchmark. No campus evaluation has been conducted yet.

1. Create `local_evaluation/` in the project and copy `templates/photo_labels.csv` there. Keep photos in that folder. This folder is excluded from Git.
2. Before looking at predictions, collect permissioned single-item photos and assign labels based on the actual material. Exclude ambiguous mixed-material items from the six-class score and record them separately. Avoid faces, personal information and private locations.
3. Use one photo per distinct physical item. Include all six classes, with a preplanned target per class. Record collection date, collector, labeling method, exclusions and any uncertainty in a local collection note. A small convenience sample does not represent the entire campus.
4. Add rows to the CSV, for example `item001,photos/bottle.jpg,plastic`. Valid labels are cardboard, glass, metal, paper, plastic and trash. Paths are relative to the CSV. Do not use the example without a real photo and independently assigned label.
5. From the project folder, run:

```powershell
.venv\Scripts\python.exe evaluate_photos.py local_evaluation/photo_labels.csv --output local_evaluation/result.json
```

Run the Windows launcher first if the project environment has not been installed. Choose a new output filename for each run. Do not change labels just to agree with predictions.

The report saves accuracy, per-class precision/recall/F1 and support, a confusion matrix (actual rows, predicted columns), per-item predictions, model identity and checksums. It omits photo paths and image bytes. Review item identifiers before sharing. Missing classes produce a warning and no six-class macro F1; the present-class average must not be compared as if it covered all six classes.

Invalid photos or labels stop the run instead of silently dropping failures and inflating accuracy. Duplicate item IDs, duplicate image pixels, repeated paths and exact matches against recorded dataset pixels are rejected. These checks cannot prove that crops or different views are independent. Similar objects, exclusions and label quality still require review.

Report this set's origin, sampling method, per-class counts, results and limitations separately from TrashNet. If these photos guide model improvements, they become development data; collect another untouched set for a later final evaluation. Campus disposal instructions require official local rules, independently of material classification.
