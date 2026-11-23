# WasteSort progress report

Updated November 23, 2026. Repository: https://github.com/haerinfan/wastesort (16 commits on `main`).

WasteSort is a six-class waste-material image classifier (cardboard, glass, metal, paper, plastic, trash). It compares four methods on the TrashNet benchmark, then tests whether benchmark accuracy holds on campus-like scenes, using a primary dataset of AI-generated images and the group's own photographs.

## Overall progress: about 78%

The percentages are our own estimates of how much of each planned deliverable exists and has been checked. They are not measured quantities. The overall figure is the weighted average of the rows below.

| Workstream | Weight | Progress | Status |
|---|---:|---:|---|
| Secondary dataset (TrashNet): download, audit, duplicate handling, grouped splits | 15% | 100% | Done |
| Primary dataset: 2,517 generated images + 11 real photos | 15% | 90% | Done; human review of the generated-image sheet still open |
| Model training and experiments (benchmark, domain shift, fine-tuning) | 20% | 100% | Done for the initial results |
| Prototype app (Streamlit, local) | 10% | 85% | Works locally; no hosting, no bin guidance |
| IMRAD research paper (ACM format, Word + PDF) | 15% | 95% | Complete draft; awaiting instructor feedback |
| Testing and reproducibility (tests, notebooks, pinned environment) | 5% | 100% | 30 automated tests pass |
| Real-world campus validation | 15% | 10% | Only 9 scored photos so far |
| Campus disposal guidance in the app | 5% | 0% | Needs official campus rules |
| **Overall** | **100%** | **≈ 78%** | |

## Implemented features

**Data**
- TrashNet download from a pinned source with checksum, image audit, exclusion of conflicting exact duplicates, and grouped 70/15/15 splits so near-duplicate photos never cross partitions ([data/splits.csv](../data/splits.csv)).
- Primary dataset: 4,541 images generated with SDXL-Turbo from a fixed prompt plan of campus-like scenes; CLIP quality check and duplicate removal kept 2,517, matched to TrashNet's class counts (plastic 476 of 480). All generated images are labelled as AI-made in the paper, datasheet and file metadata ([PRIMARY_DATASET.md](PRIMARY_DATASET.md)).
- 11 real photographs taken by the group; 9 single-item photos are used as a real-world test.

**Models**
- Four methods: HOG + SVM, HOG + Random Forest, a small CNN and MobileNetV2 (transfer learning), plus a class-weighted CNN ablation. KNN, Decision Tree and Linear Regression exercises are also included.
- Model selection uses validation macro F1 only; the test set is evaluated once, after the choice is frozen.
- MobileNetV2 fine-tuned on TrashNet + generated images.

**Application and tooling**
- Streamlit app with four tabs: Classify an item, Research results, Datasets, About the project. The fine-tuned model is available as an optional second model.
- Batch prediction script, separate-photo evaluator with a label template, model checksums and an environment check (`python -m wastesort.doctor`).
- Colab notebooks for training and for building the primary dataset; local GPU option (`--low-vram`) for 8 GB cards.
- 30 automated tests covering upload validation, features, split integrity, model selection and app behaviour.

**Documents**
- IMRAD paper in ACM format, 6 pages, 7 tables, 3 figures, 17 references ([PDF](../documents/WasteSort_ACM_Paper_Draft.pdf), [Word](../documents/WasteSort_ACM_Paper_Draft.docx)).
- Results summary, model card, datasheet, submission guide and earlier weekly presentation decks.

## Key results

| Result | Value |
|---|---|
| Best model on TrashNet test (MobileNetV2) | 84.57% accuracy, 81.22% macro F1 |
| Other methods, TrashNet macro F1 | SVM 63.52%, Random Forest 60.22%, small CNN 57.68% |
| MobileNetV2 on generated campus-like images | 59.73% macro F1 (−21.49 points) |
| Classical methods on generated images | SVM 11.37%, Random Forest 10.02% macro F1 |
| How different the datasets look | A classifier tells them apart 98.97% of the time |
| Fine-tuned MobileNetV2, generated test | 87.66% macro F1 (was 58.50%) |
| Fine-tuned MobileNetV2, TrashNet test | 82.45% macro F1 (was 81.22%, no loss) |
| Group's real photos correct | 3 of 9 before fine-tuning, 5 of 9 after |

Main finding: a high benchmark score does not carry over to everyday scenes. Adding generated scene images closes most of the gap on generated images without hurting TrashNet performance, but real-photo evidence is still small.

## Planned features and next steps

| Priority | Planned work | Why |
|---|---|---|
| 1 | Collect at least 50 real campus photos per class, labelled and split by item | The only way to measure real accuracy |
| 2 | Human review of the generated-image contact sheet | Remove wrong-material images the CLIP check missed |
| 3 | Add soft plastics (bags, film, pouches) to the prompt plan | Plastic and trash fail most often |
| 4 | Strengthen the trash class with more examples and class weighting | Lowest recall in every test |
| 5 | Fine-tune more MobileNetV2 layers; compare a larger backbone | Possible further accuracy gains |
| 6 | Show "not sure" when confidence is low (calibrated scores) | Avoid confident wrong advice |
| 7 | Add official campus bin rules to the app | Turn material labels into disposal guidance |
| 8 | Repeat training over several seeds and splits | Measure result variance |
| 9 | Pilot test with students at campus bins | Check usefulness in practice |

## Limitations to state openly

- Results come from one seed and one split.
- The generated images are AI-made; results on them measure transfer to generated scenes, not real campus accuracy.
- Nine real photos show failure cases but cannot estimate accuracy.
- The app predicts material only. It does not judge recyclability, cleanliness or hazards.
- AI tools assisted with implementation, experiment runs and writing; the group is responsible for checking and explaining the work.
