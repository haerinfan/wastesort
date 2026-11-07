# Research progress submission

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
