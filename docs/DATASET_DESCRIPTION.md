# Secondary dataset: TrashNet

Creators: Gary Thung and Mindy Yang. Source: https://github.com/garythung/trashnet
Archive: https://huggingface.co/datasets/garythung/trashnet/resolve/94cd17e/dataset-resized.zip
Retrieved October 20, 2026; source documentation checked November 7, 2026.

The creators photographed individual waste items on white posterboard with sunlight and/or room lighting using iPhone cameras. Their resized archive contains 2,527 RGB JPEGs, 512 by 384 pixels, in six material classes. This group downloaded and audited that existing archive; it did not collect those photographs. Source class counts: cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.

Six conflicting-label exact-pixel files were excluded, leaving 2,521 observations. Frozen split: 1,774 train, 371 validation, 376 test. Target: label (six material categories); label_id is cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5. Codes are categorical, not ordered measurements.

reports/dataset_inventory.csv lists all 2,527 source records and inclusion/exclusion information. Blank exclusion_reason for included rows and blank split/group for excluded rows mean not applicable, not missing model inputs. data/splits.csv contains the usable records and original hashes/group assignments.

data/secondary/Secondary_TrashNet_HOG_Features.csv contains 2,521 rows, five metadata/target columns (path,label,label_id,group,split) and 1,764 numeric HOG feature columns hog_0000 through hog_1763. These are actual features extracted from the checksum-verified images using the project implementation, not synthetic data. Metadata columns must not be used as predictors. Features describe edge orientations in 64 by 64 grayscale images. They are unscaled: SVM/KNN fit StandardScaler only on training rows. No learned feature selection was applied.

MobileNetV2 and the small CNN use the source images, not the HOG CSV. Download and extract the pinned source archive to inspect original image records; the project downloader establishes the data/raw/<class>/<filename> paths used by the CSVs. The separately supplied submission ZIP includes the raw image archive because CSV indices alone do not encode the neural model inputs. GitHub users download it from the pinned source. The runnable notebook can download the same pinned archive automatically.

Data cleaning: every source image decoded and had a known folder label; no pixel or target imputation was performed. Six files in three identical-pixel pairs had conflicting labels, and all copies were removed. Forty-five perceptual-hash candidates were grouped within partitions and later visually screened with AI assistance. No statistical outlier removal was performed; unusual appearances were retained unless covered by the documented exclusion rule. The original class imbalance was preserved. Full preprocessing and model details appear in the technical report.

No primary dataset is claimed. Dataset attribution and MIT notice are supplied in THIRD_PARTY_NOTICES.md. Do not treat public-dataset results as campus validation or automatic disposal guidance.
