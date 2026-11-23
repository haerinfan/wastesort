# Research progress submission

## Week 16 submission: IMRAD paper with initial machine learning results

Submit the [IMRAD paper PDF](../documents/WasteSort_ACM_Paper_Draft.pdf) and its [editable Word version](../documents/WasteSort_ACM_Paper_Draft.docx). It uses the ACM template and follows the five required parts. Fill in the three author e-mail addresses (highlighted in yellow) before submitting.

| Required part | Paper section |
|---|---|
| Introduction: background, objectives, significance | 1.1 Background, 1.2 Objectives (RQ1–RQ4), 1.3 Significance, 1.4 Related Work |
| Methods: dataset, preprocessing, algorithms, training | 2.1–2.7: TrashNet secondary data, real photographs and generated images (primary dataset), preprocessing, four pipelines, training, selection and evaluation |
| Results: performance, metrics, tables, visualisations | 3.1–3.5: Tables 2–7 and Figures 1–3 (learning curves, confusion matrix, dataset samples) |
| Discussion: interpretation, limitations, improvements | 4.1 Interpretation of Findings, 4.2 Limitations and Ethics, 4.3 Recommended Improvements |
| Conclusion and references | 5 Conclusion and Future Work, 17 references in ACM style |

| Submission requirement | Evidence |
|---|---|
| IMRAD manuscript (editable and PDF) | [WasteSort_ACM_Paper_Draft.docx](../documents/WasteSort_ACM_Paper_Draft.docx), [WasteSort_ACM_Paper_Draft.pdf](../documents/WasteSort_ACM_Paper_Draft.pdf); build scripts in [paper_build](../documents/paper_build/) |
| Source code | [wastesort/](../wastesort/), [app.py](../app.py), [training notebook](../notebooks/WasteSort_Colab.ipynb), [primary-dataset notebook](../notebooks/WasteSort_Primary_Colab.ipynb) |
| Dataset or access documentation | Secondary: [dataset description](DATASET_DESCRIPTION.md), [splits.csv](../data/splits.csv), [provenance](../data/provenance.json) (images download from the pinned URL). Primary: [datasheet](PRIMARY_DATASET.md), [real photographs](../data/primary_set/real_photos/), [2,517 generated images](../data/primary_set/images/), [splits](../data/primary_set/splits.csv), [all candidates with CLIP scores](../data/primary_set/candidates.csv) |
| Initial training results | [Results summary](RESULTS.md), [TrashNet test evaluation](../reports/test_evaluation.json), [model comparison](../reports/comparison.csv), [primary-set results](../reports/primary_set/), trained models in [models/](../models/) |
| Performance visualisations | Learning curves ([MobileNetV2](../reports/mobilenet_learning.png), [CNN](../reports/cnn_learning.png)), confusion matrices ([MobileNetV2](../reports/mobilenet_confusion.png), [fine-tuned](../reports/primary_set/confusion_mobilenet_primary_on_primary_test.png)), [dataset samples](../reports/primary_set/sample_grid.png), [PCA of features](../reports/primary_set/embedding_pca.png), [image statistics](../reports/primary_set/image_statistics_comparison.png) |

All reported numbers come from actual runs: the TrashNet benchmark (October 20, 2026) and the primary-dataset experiments on an NVIDIA RTX 2080 (November 23, 2026). The generated images are AI-made and disclosed as such in the paper, the datasheet and each file's metadata. For the presentation, the key findings are: MobileNetV2 is best on TrashNet (81.22% macro F1) but recognises 3 of 9 real photographs; it drops to 59.73% on generated campus scenes; adding generated training images raises that to 87.66% without losing TrashNet accuracy and gives 5 of 9 real photographs.

## Week 14 submission (historical)

The Week 14 section below describes the earlier secondary-data-only package. The primary dataset described above was added afterwards and supersedes its statement that no primary data were collected.

Use the [Introduction and Methodology PDF](../documents/Goal_Diggers_WasteSort_Introduction_and_Methodology.pdf) or its [editable Word version](../documents/Goal_Diggers_WasteSort_Introduction_and_Methodology.docx). The [manuscript source](WEEK14_MANUSCRIPT.md) covers all six Introduction topics and seven Methodology topics in the November 3–7 instructions, with 13 references. This is the current manuscript; older proposal documents are historical and are superseded where terminology or implementation differs. In particular, TrashNet is secondary data, even where an older proposal called it primary.

Submit the manuscript together with the datasets, trained models in `models/`, source code, and preliminary outputs mapped below. The separate `Goal_Diggers_Week14_Submission.zip` supplied in chat includes the source image archive and a current project snapshot. Results and Discussion, conclusions, and the full IMRAD paper remain the Week 16 deliverable for November 16–21, 2026.

The supplied checklist permits primary **and/or** secondary data. This package supplies authentic secondary data; no uncollected primary dataset is represented as complete.

This is a **secondary-data study**. The observations are TrashNet images collected by Gary Thung and Mindy Yang, downloaded and prepared for WasteSort. No primary observations or campus collection are claimed. Derived image features remain derivatives of secondary data. If primary collection is required by the instructor, that requirement remains outstanding.

| Requirement | Submitted evidence |
|---|---|
| Dataset description, source and collection method | [Dataset description](DATASET_DESCRIPTION.md), [source provenance](../data/provenance.json), [source and study scope PDF](../documents/Dataset_Source_and_Study_Scope.pdf) |
| CSV datasets | [All 2,527 source records](../reports/dataset_inventory.csv), [2,521 retained records and frozen split](../data/splits.csv), [actual HOG features and labels](../data/secondary/Secondary_TrashNet_HOG_Features.csv) |
| Cleaning and preprocessing | [Technical report](../documents/Brief_Technical_Report.pdf), [preprocessing record](../reports/preprocessing.json), [exclusions](../reports/excluded.csv), [duplicate review](DUPLICATE_REVIEW.md) |
| Source code and actual training | [Runnable full notebook](../notebooks/WasteSort_Colab.ipynb), [executed full notebook](../notebooks/WasteSort_Colab_Executed.ipynb), [Python training source](../wastesort/train.py) |
| Initial training results PDF | [Initial Model Training Results](../documents/Initial_Model_Training_Results.pdf) |
| Methodology, findings and recommendations | [Brief Technical Report](../documents/Brief_Technical_Report.pdf) |

The feature CSV contains 2,521 rows, five metadata/target columns and 1,764 numeric HOG features. Do not use paths, labels, split or group fields as predictors. The CSV reproduces the original SVM test accuracy (248/376 correct). The neural models consume image tensors rather than this HOG table.

The original resized images are downloaded by the runnable notebook from the pinned URL in source provenance; they are not stored directly in Git. The separate submission ZIP supplied in chat includes that original archive as well. GitHub's Code > Download ZIP includes the reports, CSVs, source and trained checkpoints but requires the documented image download for retraining.

The PDFs report the October 20 Windows benchmark. The saved full Colab notebook records a separate October 26 run; CNN differences are described in [Colab verification](COLAB_VERIFICATION.md). Prepared November 7, 2026. Presentation delivery, instructor acceptance and campus evaluation are not implied by file availability.
