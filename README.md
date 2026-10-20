# WasteSort

## Start here: Goal Diggers learning materials

The **Monday activity** presentation by **Goal Diggers** is ready as a short, approximately four-minute explanation of AI, ML, learning types, practical applications and supervised image classification in WasteSort:

- [PowerPoint with speaker notes](presentation/WasteSort_Monday_Concepts.pptx)
- [PDF slide backup](presentation/WasteSort_Monday_Concepts.pdf)
- [Speaking guide and Q&A](presentation/WasteSort_Monday_Speaking_Guide.pdf)
- [Brief written explanation](presentation/Monday_Brief_Explanation.md)

These materials focus on the assigned concepts. The broader implementation and experiment results below support later project work.

### Day 2: Tuesday dataset preparation

- [PowerPoint with speaker notes](presentation/day2/WasteSort_Tuesday_Dataset_Preparation.pptx)
- [PDF slides](presentation/day2/WasteSort_Tuesday_Dataset_Preparation.pdf)
- [Speaking guide, demo steps and Q&A](presentation/day2/WasteSort_Tuesday_Speaking_Guide.pdf)
- [Brief explanation](presentation/day2/Tuesday_Brief_Explanation.md)
- [Runnable preparation notebook](notebooks/WasteSort_Day2_Tuesday.ipynb)
- [Executed notebook with saved outputs](notebooks/WasteSort_Day2_Tuesday_Executed.ipynb)
- [Complete source-image inventory](reports/dataset_inventory.csv), [label encoding](reports/label_encoding.json), [preprocessing plan](reports/preprocessing.json) and [human review worksheet](reports/manual_review_worksheet.csv)

The Tuesday notebook uses a CPU runtime and stops at prepared inputs. In Colab select **Runtime → Run all**. Its executed copy was verified locally on Windows; Google Colab execution remains unverified. The worksheet intentionally leaves human decisions blank.

## Project

A six-class waste-material research prototype. Upload one item photo in the Streamlit app to predict **cardboard, glass, metal, paper, plastic, or trash**.

The project compares HOG + SVM, HOG + Random Forest, a small CNN and MobileNetV2. KNN and Decision Tree are introductory exercises; a small-CNN class-weighting ablation and a separate numerical regression exercise are also included.

**This is an initial benchmark study, not a validated campus disposal system.** Campus bin rules, human image review and group verification remain pending. The app can label unrelated objects as waste and does not detect hazards or contamination.

## Run the app

Use **Python 3.12**. A trained deployment checkpoint is included; running the app does not require downloading the dataset or retraining.

```powershell
git clone https://github.com/haerinfan/wastesort.git
cd wastesort
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. Open the local address printed by Streamlit. Upload a JPEG/PNG and select **Identify material**.

For Windows NVIDIA GPU training, install the matching PyTorch build before the other requirements:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

CPU inference works without CUDA. All package installation is isolated in the virtual environment.

## Results and research record

- [Measured results](docs/RESULTS.md)
- [Dataset audit](reports/data_audit.json) and [class/split counts](reports/split_counts.csv)
- [Excluded files and reasons](reports/excluded.csv)
- [Frozen image manifest](data/splits.csv)
- [Protocol fixed before training](protocol.json)
- [Model card and limitations](docs/MODEL_CARD.md)
- [Document review and changes](docs/REVIEW_AND_CHANGES.md)
- [Revised proposal and weekly status](documents/)
- [Training environment](reports/environment.json)
- [Same-environment reproduction check](reports/reproduction_check.json)

The deployment choice is recorded before test evaluation in `models/deployment.json`. Training uses approximately 70/15/15 grouped partitions and seed 42. Settings are selected with validation macro F1. The test images are evaluated only after checkpoints and the deployment choice are frozen.

Exact pixel duplicates with inconsistent labels are excluded in full. Perceptual-hash candidate groups stay in one partition. Automated checks cannot guarantee that every repeated physical object or incorrect label has been found.

## Reproduce the training

Open [WasteSort_Colab.ipynb](notebooks/WasteSort_Colab.ipynb) in Google Colab. Select **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. The notebook downloads real data, creates a fresh experiment directory and runs the pipeline. It does not overwrite the published results.

Alternatively, from an installed local environment:

```powershell
python scripts/rerun.py
```

This creates a new directory under `runs/` and runs:

1. Download the creators' **resized archive only**, verify its checksum, audit all images and reproduce the manifest.
2. Train four primary methods, two introductory classifiers and the CNN weighting ablation.
3. Freeze the validation-selected deployment model.
4. Evaluate the five research checkpoints once on the held-out test images.
5. Save predictions, metrics, confusion matrices, learning curves and descriptive bootstrap intervals.
6. Run the separate regression exercise.

The pipeline refuses to retrain over an experiment that already contains final test results. Use a fresh run for reproduction. Do not select a favorable seed or revise the model based on published test errors while presenting the same test as unseen.

`WASTESORT_ROOT` can point to a fresh experiment directory containing `protocol.json`. The source code remains in this repository; data, checkpoints and reports are written to that directory.

## Work with groupmates

Share **https://github.com/haerinfan/wastesort**. Anyone can view or clone this public repository. Write access requires the owner to add the person's GitHub username under **Settings → Collaborators**.

Use one branch per task and a pull request for review:

```text
git switch -c yourname/task-name
# make and check changes
git add <changed-files>
git commit -m "Describe the completed work"
git push -u origin yourname/task-name
```

Record real contributions in [CONTRIBUTIONS.md](CONTRIBUTIONS.md). Suggested next tasks are in [GROUP_CHECKLIST.md](docs/GROUP_CHECKLIST.md). AI-generated work and automated experiment execution must not be reported as independent member completion.

## Tests

```powershell
python -m pytest -q
```

Tests cover upload validation, feature dimensions, candidate grouping, split integrity, consistent preprocessing, validation-only model selection and application behavior. Read [verification](docs/VERIFICATION.md) for the checks actually run.

## Data and attribution

TrashNet was created by **Gary Thung and Mindy Yang**. Source: [original repository](https://github.com/garythung/trashnet), [creators' dataset host](https://huggingface.co/datasets/garythung/trashnet).

The original source reports 2,527 images. Download metadata, a pinned source revision and the archive SHA-256 are in `data/provenance.json`. Both original and resized archives exist upstream; do not combine them.

Dataset images are downloaded on demand and excluded from Git. The creators' MIT notice is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). MobileNetV2 uses torchvision's ImageNet pretrained weights; see the model card for attribution and restrictions on interpretation.

## AI assistance

OpenAI ChatGPT/Codex assisted with review, implementation, data auditing, training execution, debugging and documentation. The group must independently inspect, reproduce and explain the study, verify references, confirm the revised protocol with the instructor and record actual contributions.

No public app hosting is configured. The public GitHub repository shares the code, notebook, evidence and deployed model; the app runs locally.
