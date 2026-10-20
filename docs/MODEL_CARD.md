# WasteSort model card

**Version:** initial benchmark, October 20, 2026.
**Purpose:** educational six-class material recognition from one item photograph.

## Model and input

The deployed model uses torchvision MobileNetV2 with `IMAGENET1K_V2` pretrained weights. Its backbone, including batch-normalization statistics, is frozen; a dropout layer and six-output linear head were trained on TrashNet. This is transfer learning, not training the whole network from scratch.

Inputs are JPEG/PNG images up to 10 MB and 20 million pixels, with each side at least 32 pixels. Images are oriented using EXIF, converted to RGB, resized to 160 × 160 and normalized with ImageNet mean and standard deviation. Training-only augmentation is specified in [protocol.json](../protocol.json).

Outputs are **cardboard, glass, metal, paper, plastic, trash**. Softmax scores are uncalibrated outputs. A score is not a verified probability of correctness. The 0.60 low-score message is a usability cue, not a validated rejection threshold.

## Training data and evaluation

The source is [TrashNet by Gary Thung and Mindy Yang](https://github.com/garythung/trashnet). The creators' resized archive is downloaded from their pinned dataset-host revision. See [provenance](../data/provenance.json), [manifest](../data/splits.csv) and [audit](../reports/data_audit.json).

After exclusion of six conflicting duplicate files, 2,521 images were partitioned into 1,774 training, 371 validation and 376 test images. Exact-pixel and candidate near-duplicate checks were applied before splitting. Human label and grouping review remains required.

The model was selected using validation macro F1, then evaluated once on the held-out test split. Test accuracy was 84.57%, macro F1 81.22%, and trash recall 55.56% (10/18). See [results](RESULTS.md) for all comparisons and uncertainty.

## Appropriate use and limitations

Use for classroom demonstrations and research comparisons under conditions similar to the source images. Do not use it to decide whether a material is safe, clean, recyclable, hazardous or allowed in a particular campus bin. Campus guidance requires separately verified institutional rules.

The model always selects one of six labels; unrelated objects may receive confident waste labels. It does not locate multiple objects, recognize all material types or measure weight. Composite packaging, cluttered scenes, local lighting and different cameras may reduce accuracy. The small trash test sample makes its recall estimate uncertain.

The application processes uploads in memory and does not explicitly save them. No public application hosting is configured. Any later hosting must separately assess access, infrastructure logging and upload handling.

## Attribution and maintenance

TrashNet's MIT notice is reproduced in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Model architecture and pretrained weights come from [torchvision](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v2.html); attribution is not evidence of suitability for every downstream use.

The deployment checkpoint and SHA-256 are recorded in [deployment.json](../models/deployment.json). The public repository includes the deployed checkpoint; other experiment checkpoints can be reproduced with the notebook or local runner. Only load trusted model files.

AI assisted implementation and experiment execution. Group members must review labels, reproduce the study, verify sources and explain the results. Instructor acceptance, individual completion and campus validation are not certified by this model card.
