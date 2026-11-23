# Progress presentation script

For the November 24, 2026 progress report. Use the [progress report](PROGRESS_REPORT.md) as the handout. Speakers 1–3 can be reassigned; each part is self-contained.

Have these open before starting:
- The GitHub repository page: https://github.com/haerinfan/wastesort
- [docs/PROGRESS_REPORT.md](PROGRESS_REPORT.md) on GitHub
- The paper PDF: [documents/WasteSort_ACM_Paper_Draft.pdf](../documents/WasteSort_ACM_Paper_Draft.pdf)
- Optional: the app running locally (`python -m streamlit run app.py`) with one photo ready to upload
- If asked to run the tests live, use an environment installed from `requirements.txt` (scikit-learn 1.6.1). With a newer scikit-learn, one split-reproduction test fails because the library shuffles groups differently, even though the saved splits are unchanged.

---

## Short version (about 2 minutes)

Good morning, sir/ma'am. We are presenting the progress of WasteSort, an image classifier that identifies the material of a waste item: cardboard, glass, metal, paper, plastic or trash.

We are at about 78% overall. The data, the models, the experiments and the IMRAD paper are done. What remains is mainly real-world validation on campus.

We trained four methods on the TrashNet benchmark. The best is MobileNetV2, with 84.6% accuracy and 81.2% macro F1. But when we tested on our primary dataset, 2,517 AI-generated images of campus-like scenes, it dropped to 59.7% macro F1. On nine of our own photos it was right only 3 times. So a good benchmark score does not mean the model works on everyday photos.

When we added the generated images to training, macro F1 on that set rose to 87.7%, TrashNet performance stayed the same, and our real photos went from 3 to 5 correct.

Everything is on GitHub: the code, the datasets, the results and the paper, with 30 automated tests. Next we will collect real campus photos, add soft plastics, strengthen the trash class and add campus bin rules to the app. Thank you.

---

## Long version (about 8–10 minutes)

### Part 1 – Introduction and progress (Speaker 1, about 2 minutes)

*[Show the repository home page.]*

Good morning, sir/ma'am. We are Goal Diggers, and our project is WasteSort.

People often put waste in the wrong bin, which lowers the value of recyclables. Our idea is a camera-based helper: take a photo of one item and the system tells you its material, one of six classes: cardboard, glass, metal, paper, plastic or trash.

Our research question is simple: if a model scores well on a public benchmark, does it still work on the kinds of photos students actually take?

*[Open docs/PROGRESS_REPORT.md and scroll to the progress table.]*

Our overall progress is about 78%. These percentages are our own estimates per deliverable. The datasets, model training, experiments and testing are complete. The app works locally. The IMRAD paper is a complete draft. The parts still low are real campus validation, at 10%, and campus disposal rules in the app, which we have not started because we need official rules first.

### Part 2 – Data and methods (Speaker 2, about 3 minutes)

*[Scroll to "Implemented features", or show the commit history.]*

We used two datasets.

The secondary dataset is TrashNet, a public dataset of about 2,500 photos, mostly single items on white backgrounds. We audited it, removed conflicting duplicates and split it 70/15/15. Near-duplicate photos stay in the same split, so the model is never tested on a copy of a training image.

The primary dataset has two parts. First, 11 real photos taken by our group; 9 single-item photos are used as a real-world test. Second, AI-generated images. With our instructor's approval we generated 4,541 images of waste items in campus-like scenes: outdoor bins, tables, leaves. An automatic CLIP check removed images that did not match their label, and we kept 2,517, matched to TrashNet's class counts. We disclose clearly that these images are AI-generated, in the paper, the datasheet and every file.

We compared four methods: HOG features with an SVM, HOG with a Random Forest, a small CNN we built ourselves, and MobileNetV2, a pretrained network we fine-tuned. Models are chosen on the validation set only, and the test set is used once at the end, so the test results are not tuned.

### Part 3 – Results (Speaker 3, about 3 minutes)

*[Open the paper PDF to the results tables, or the app's Research results tab.]*

On TrashNet, MobileNetV2 is clearly best: 84.6% accuracy and 81.2% macro F1. The SVM got 63.5%, the Random Forest 60.2% and the small CNN 57.7% macro F1.

Then we tested the same models on the generated campus-like images. The two datasets look very different: a simple classifier can tell them apart 99% of the time. MobileNetV2 dropped to 59.7% macro F1, and the SVM and Random Forest fell to about 10%, close to guessing. On our nine real photos, MobileNetV2 was correct only 3 times.

Next we fine-tuned MobileNetV2 on TrashNet plus the generated images. On the generated test set, macro F1 went from 58.5% to 87.7%. On TrashNet it stayed at 82.5%, so it lost nothing. On our real photos it improved from 3 to 5 out of 9.

The hardest classes are plastic and trash. Soft plastics like bags and wrappers fail in every test, and trash is the smallest and most mixed class.

*[Optional demo: Classify tab, upload one photo, show the top predictions and confidence.]*

### Part 4 – Limitations, next steps and closing (Speaker 1, about 1–2 minutes)

*[Scroll to "Planned features and next steps".]*

We want to be honest about the limits. These are results from one seed and one split. Results on generated images are not real campus accuracy, and nine photos are too few to measure accuracy. The app identifies material only; it does not check if an item is clean, recyclable or hazardous.

Our next steps, in order: collect at least 50 real campus photos per class; have people review the generated images; add soft plastics; strengthen the trash class; show "not sure" when confidence is low; and add the official campus bin rules.

All of our work is on GitHub: source code, notebooks, both datasets with access documentation, trained models, results and the IMRAD paper in Word and PDF. We also have 30 automated tests. Thank you, and we welcome your questions.

---

## Likely questions

**Why use AI-generated images as primary data?**
Collecting about 2,500 labelled campus photos was not possible in the time we had. The instructor approved generated images as primary data. We disclose them, filtered them with CLIP, and kept real photos as a separate test.

**Isn't 87.7% on generated images misleading?**
It measures how well the model adapts to generated scenes, not real campus accuracy. That is why we report the real-photo result separately and list real photo collection as the first next step.

**Why MobileNetV2?**
It is pretrained on ImageNet, so it already knows general visual features. It is small enough for a laptop and gave the best validation score.

**What is macro F1, and why not just accuracy?**
Macro F1 averages the F1 score of each class equally. Trash is a small class, so accuracy alone could look good while trash is almost never recognised.

**How do we know the test set was not used for tuning?**
Model choice used validation macro F1 only. The choice was saved before the test evaluation, and the pipeline refuses to retrain over finished test results.

**What did AI tools do?**
They assisted with code, experiment runs and writing. We checked the results and can explain each step; all numbers come from actual runs saved in the repository.

**Can it tell us which bin to use?**
Not yet. It predicts material only. Bin guidance needs official campus rules, which is a planned feature.
