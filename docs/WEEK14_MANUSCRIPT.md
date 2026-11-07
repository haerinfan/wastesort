# WasteSort Supervised Image Classification of Common Waste Materials for Campus Recycling Guidance

Introduction and Methodology Manuscript

Goal Diggers

Lordy Allen D. Reyes • Josh Patrick R. Reyes • John Benjamin S. Tagbo

De La Salle University–Dasmariñas

College of Information and Computer Studies

BCS34 • S-CSIS311 Introduction to Machine Learning

Week 14 • November 7, 2026

## 1 Introduction

### 1.1 Background of the Study

WasteSort investigates whether supervised image classification can identify the dominant material in a photograph of a single waste item. The intended application is an educational decision-support tool for campus recycling. Recognizing a material is a preliminary step: the appropriate disposal route also depends on contamination, local collection arrangements, and campus policies. Accordingly, the implemented prototype predicts one of six material categories and does not establish whether an item is locally recyclable.

The study uses TrashNet, an existing collection of labeled photographs of cardboard, glass, metal, paper, plastic, and miscellaneous trash. Its creators photographed items on white posterboard under sunlight or room lighting (Thung & Yang, n.d.). These controlled photographs make a manageable starting point for an undergraduate classification study, while also limiting conclusions about cluttered scenes and everyday campus photography. The research question is therefore bounded to a reproducible public-dataset experiment before any claim of campus effectiveness.

Supervised learning learns a mapping from labeled examples to target categories. For WasteSort, the predictors are either numerical descriptors extracted from images or RGB pixels supplied to a neural network; the target is a categorical material label. A model learns on training images, development decisions use validation images, and a separate test partition estimates performance after model selection. This separation is necessary because information leaking between development and evaluation can produce misleading performance estimates (Kapoor & Narayanan, 2023).

The project compares two feature-based classifiers with a small convolutional neural network and a pretrained MobileNetV2. The comparison examines the practical choice between engineered visual descriptors and learned representations. It also supplies an initial trained model for a working upload interface, linking machine-learning methodology with a concrete computer-science application.

### 1.2 Research Gap

Waste recognition is an established research topic; this study does not claim to introduce it. TrashNet provides isolated-item photographs, whereas TACO was developed for litter detection and segmentation in contextual scenes (Proença & Simões, 2020). These datasets address different conditions and tasks. Performance on an isolated-item classifier cannot, by itself, answer whether the same system will work on multiple objects, campus bins, or unfamiliar backgrounds.

The specific gap addressed by WasteSort is the need for a traceable comparison within the group's intended application: identical usable records and partitions, documented duplicate handling, validation-based selection, and explicit reporting of minority-class performance. A single accuracy score would not adequately describe performance on the relatively small trash category. The contribution is a reproducible experimental pipeline and a limited prototype, rather than a new architecture or a demonstrated improvement over all published waste-recognition systems.

Campus generalization remains an unresolved extension. It requires independently collected local photographs and verified disposal rules. Neither is represented by the current secondary dataset. The manuscript therefore separates the completed public-dataset methodology from that future validation requirement.

### 1.3 Statement of the Problem

The central problem is to determine how accurately and consistently the selected learning approaches distinguish six waste-material categories under a documented evaluation procedure. Similar appearances across materials and unequal class frequencies may obscure failures when performance is summarized only by overall accuracy.

The study asks the following research questions:

1. How do an SVM and a Random Forest using HOG features compare with a small CNN and frozen-backbone MobileNetV2 on the same grouped TrashNet partitions?
2. Which candidate achieves the highest validation macro F1, and what are its held-out accuracy, class-level precision, recall, F1, and confusion patterns?
3. How does class-weighted cross-entropy affect the small CNN, particularly its recognition of the trash category, under otherwise matched settings?
4. Can the selected checkpoint be integrated into a single-image upload prototype with consistent preprocessing and clearly stated limits?

These questions are answered descriptively for one fixed split and seed. The experiment is not designed to establish statistical superiority across populations or to measure campus waste-segregation behavior.

### 1.4 Research Objectives

The general objective is to develop and evaluate a reproducible supervised waste-image classifier and an educational prototype that can support later investigation of campus recycling guidance.

The specific objectives are to:

1. Audit all 2,527 source images, record exclusions and provenance, and freeze group-separated training, validation, and test records.
2. Train four research candidates using documented features, preprocessing, hyperparameters, and checkpoint-selection rules.
3. Select the deployment candidate by validation macro F1 across the six fixed labels, then evaluate its frozen checkpoint and the comparison models on the test partition.
4. Report overall and per-class metrics, confusion matrices, learning histories, and descriptive uncertainty intervals, with a separate weighted-CNN comparison.
5. Deliver the dataset tables, source code, trained artifacts, and an upload application that uses the selected model's evaluation preprocessing.

### 1.5 Significance of the Study

For students and instructors, WasteSort provides an inspectable example of dataset auditing, feature extraction, training, evaluation, and deployment. Its saved partitions and predictions make it possible to explain how an individual result was obtained rather than relying only on a demonstration screen.

For potential campus users, the prototype illustrates how a material-recognition interface could assist learning about waste categories. Any reduction in sorting mistakes, improvement in recycling rates, or user benefit would require a separate field study; these outcomes are not measured here. For subsequent researchers, the documented exclusions and grouped evaluation offer a basis for replication and for designing a more representative local dataset.

### 1.6 Review of Related Literature

Waste-image datasets define the scope of the task. TrashNet supplies six material labels for predominantly isolated objects (Thung & Yang, n.d.). TACO instead supplies contextual litter annotations for detection and segmentation (Proença & Simões, 2020). Their contrast informs the present scope: WasteSort classifies one image into one material category and does not locate individual objects in a scene. TACO is discussed as related work and is not included in the training data.

Feature-based learning provides useful comparison models. Histograms of Oriented Gradients describe local edge-orientation structure (Dalal & Triggs, 2005). Although that work concerned human detection, the descriptor can be evaluated here as a general image representation without assuming that its original performance transfers to waste images. Support-vector networks provide a margin-based classification approach (Cortes & Vapnik, 1995), while Random Forests aggregate randomized decision trees (Breiman, 2001). Their different decision mechanisms motivate comparison on the same HOG representation.

Learned visual representations provide a complementary approach. ImageNet established a large labeled image resource (Deng et al., 2009), and MobileNetV2 introduced inverted residual blocks and linear bottlenecks for efficient visual models (Sandler et al., 2018). WasteSort uses ImageNet-pretrained MobileNetV2 weights as a frozen feature extractor and trains a new six-class head. It compares this transfer-learning approach with a small CNN trained from scratch. The comparison does not isolate architecture alone because pretraining supplies additional prior information.

Training design also affects the comparison. Shorten and Khoshgoftaar (2019) describe geometric and color transformations as image augmentation methods. WasteSort applies a restricted set only during neural-network training. AdamW separates weight decay from the adaptive gradient update (Loshchilov & Hutter, 2019); it is the optimizer used for the trainable neural parameters. These choices are documented rather than treated as evidence that any particular architecture must perform best.

Evaluation requires more than a high aggregate score. Sokolova and Lapalme (2009) analyze classification measures and their behavior across task settings. WasteSort emphasizes macro F1 to give each of its six categories equal importance, while retaining class support and recall to make minority-class failures visible. Kapoor and Narayanan (2023) motivate careful control of leakage and reproducibility. Guo et al. (2017) show why neural confidence scores should not automatically be interpreted as reliable probabilities of correctness. Together, this literature supports grouped partitions, frozen evaluation decisions, and cautious interpretation of the application's displayed scores.

## 2 Methodology

### 2.1 Research Design

The study uses a quantitative, experimental, and comparative design for supervised multiclass image classification. Its unit of observation is a labeled source image; near-duplicate image groups are the partitioning and bootstrap units. All four research candidates use the same included records and split assignments. The comparison evaluates complete pipelines, including their different representations and pretraining, rather than attributing every difference solely to the classifier.

The implemented workflow is source acquisition and verification, image auditing, group formation, partition freezing, feature preparation, training, validation-based selection, test evaluation, and prototype integration. Classical estimators use Scikit-learn, a Python machine-learning library described by Pedregosa et al. (2011). Neural models use PyTorch and Torchvision. Data tables and summaries are processed with Pandas and NumPy.

The original recorded run used Python 3.12.14 on Windows with an NVIDIA GeForce RTX 2080, PyTorch 2.7.1 with CUDA 11.8, Torchvision 0.22.1, Scikit-learn 1.6.1, and Scikit-image 0.25.2. Seed 42 controls the documented random operations. The source, environment record, protocol, manifests, model checksums, and prediction files are retained in the repository. A separately executed Colab notebook provides additional execution evidence; its neural results are kept distinct from the original run because hardware and numerical differences can affect training.

### 2.2 Dataset Description

The study uses secondary data only. TrashNet was obtained from the creators' public dataset, using a pinned resized archive hosted at the revision recorded in the source-provenance file. The group downloaded the archive on October 20, 2026; it did not take the source photographs. The archive contains 2,527 RGB JPEG images at 512 by 384 pixels. No surveys, participant records, or researcher-collected campus photographs are included.

The source was photographed by its creators using iPhone cameras with white posterboard and available sunlight or room lighting (Thung & Yang, n.d.). This is a convenience image collection rather than a probability sample of campus waste. The study retains the six supplied material categories. After excluding six conflicting-label exact-pixel files, 2,521 usable images remain.

Table 1 shows the original and retained class distributions and frozen partition counts.

| Material | Source | Usable | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| Cardboard | 403 | 403 | 287 | 57 | 59 |
| Glass | 501 | 498 | 352 | 73 | 73 |
| Metal | 410 | 409 | 281 | 65 | 63 |
| Paper | 594 | 594 | 411 | 90 | 93 |
| Plastic | 482 | 480 | 343 | 67 | 70 |
| Trash | 137 | 137 | 100 | 19 | 18 |
| Total | 2527 | 2521 | 1774 | 371 | 376 |

The target variable is label. Its numerical encoding follows the fixed order cardboard, glass, metal, paper, plastic, trash, numbered zero through five. These codes are category identifiers and have no ordinal meaning. Neural predictors are resized RGB pixels. Classical predictors are 1,764 HOG values per image. The submitted feature CSV contains 2,521 records and 1,769 columns: path, label, label_id, group, split, and 1,764 numeric features. Paths, labels, group identifiers, and split names are metadata or targets and are excluded from predictor inputs.

The image inventory documents every source record, and the frozen-split CSV documents the usable subset. CSV tables accompany the original image archive because a feature table or file index cannot substitute for the RGB inputs used by the neural models. Source attribution, archive verification, and license notices accompany the data.

### 2.3 Data Preprocessing

Every source file was decoded and checked for a known class label. Pixel and target imputation were unnecessary. Blank exclusion reasons for retained images and blank split assignments for excluded images denote inapplicable metadata, not missing model features. Statistical outlier removal was not applied: unusual but valid visual appearances were retained rather than discarded for being difficult examples.

Exact decoded-pixel comparison found three pairs with contradictory labels. All six files were excluded: glass115 and metal91, glass176 and plastic152, and glass389 and plastic332. Excluding both members avoids arbitrarily choosing a correct label without independent evidence. A 64-bit perceptual hash with Hamming distance at most six linked potential near duplicates; transitive links formed 2,476 groups among the 2,521 included images. Forty-five candidate pairs were identified. A later AI-assisted visual screen recorded 28 likely same-item pairs, 11 visibly distinct pairs, and six uncertain pairs. This screen was not an independent human audit, and the original grouping was retained after test results had been exposed.

Grouping is a conservative leakage-control heuristic, not proof that every repeated physical item has been identified. StratifiedGroupKFold generated 20 shuffled folds with seed 42. Folds zero through 13 form training, 14 through 16 validation, and 17 through 19 test. Related images remain within one partition. This produces 1,774 training, 371 validation, and 376 test images, approximately 70.37%, 14.72%, and 14.91%. The 20 folds construct a single holdout design; this is not 20-fold model-performance cross-validation.

For HOG, images are converted to grayscale, resized to 64 by 64, and represented on a zero-to-one pixel scale. The descriptor uses nine orientations, eight-by-eight-pixel cells, two-by-two-cell blocks, and L2-Hys normalization. SVM and the auxiliary KNN use StandardScaler fitted only on training features; the Random Forest receives unstandardized HOG features. No learned feature selection is performed.

Neural inputs are RGB images resized to 160 by 160 pixels. Training-only augmentation uses a horizontal flip with probability 0.5, rotation within ten degrees with white fill, and brightness and contrast variation of 0.1. The CNN uses zero-to-one tensors. MobileNetV2 additionally uses ImageNet channel means of 0.485, 0.456, and 0.406 and standard deviations of 0.229, 0.224, and 0.225. Validation, test, and application inference omit random augmentation. The original class imbalance is retained rather than changing evaluation frequencies through oversampling.

### 2.4 Algorithm Selection

The four research candidates represent two feature-based and two neural pipelines. The SVM uses the radial-basis-function kernel, balanced class weights, and C values of one and ten; the remaining kernel settings follow the recorded Scikit-learn defaults. The Random Forest uses 300 trees, balanced class weights, and maximum-depth candidates of unrestricted and 20. Both evaluate the same HOG descriptor, making their feature representation comparable.

The small CNN has three convolution, ReLU, and max-pooling blocks with 16, 32, and 64 channels. A five-by-five average pool converts its 20-by-20 feature map to four by four, followed by flattening, dropout of 0.3, and a six-output linear layer. For the fixed input size, this implements the four-by-four spatial pooling described in the protocol. It learns from the training partition without external pretrained weights.

MobileNetV2 uses Torchvision IMAGENET1K_V2 weights. Its backbone parameters and batch-normalization statistics are frozen; the replacement classifier comprises dropout of 0.2 and a six-output linear layer. Thus, the experiment trains a classification head rather than fine-tuning the full backbone. This approach evaluates whether pretrained representations are useful with the available dataset and training budget.

A separate small-CNN ablation compares ordinary and class-weighted cross-entropy. Each class weight equals the training-set size divided by six times that class's training count. The ablation uses the same initialization seed, split, optimizer, augmentation policy, batch order generator, epoch budget, and stopping rule. It is excluded from the four-model deployment selection. KNN and Decision Tree exercises and the separate diabetes regression exercise remain instructional artifacts outside the central WasteSort research comparison.

### 2.5 Model Development and Training

Classical models are fitted on the training partition and their limited hyperparameter candidates are compared using validation macro F1. A strict improvement replaces the current best model, so the first candidate retains an exact tie. The fitted preprocessing and estimator are saved together for consistent inference.

Neural training uses AdamW with a learning rate of 0.001, weight decay of 0.0001, batch size 32, and cross-entropy loss. The maximum budget is 20 epochs, with early stopping after five consecutive epochs without a strict validation macro-F1 improvement. The highest-scoring epoch is restored and saved. Ordinary cross-entropy is used for the principal CNN and MobileNetV2; only the specified CNN ablation uses the class weights. Training loss and validation metrics are saved for every executed epoch.

After each research candidate's checkpoint is fixed, deployment is chosen by the highest validation macro F1 across the six labels. An exact inter-model tie is resolved by lexicographic model identifier. The deployment record is frozen before test evaluation, and the test partition is not used to choose hyperparameters, stopping epochs, or the application model. Since the test results are now available, further test-driven adjustments would require a new evaluation design and must not be described as untouched testing.

The application loads the selected checkpoint after checking its stored checksum. A JPEG or PNG upload is decoded, converted to RGB, and processed with the selected model's deterministic inference transforms. The interface displays a predicted material and model scores where available. It has no verified campus-specific disposal-rule mapping, no calibrated probability guarantee, and no general out-of-distribution rejection mechanism. The intended operating case is one visible item within the six-category scope.

### 2.6 Model Evaluation

The evaluation reports accuracy and per-class precision, recall, and F1, together with macro averages, weighted averages, support counts, and confusion matrices. Accuracy is the proportion of correct predictions. For a given class, precision describes how often predictions of that class are correct; recall describes how many actual examples of that class are recovered. F1 is the harmonic mean of precision and recall. Macro F1 averages the six class F1 values equally, while the weighted average uses class support. Undefined divisions are assigned zero consistently with the implementation.

Validation macro F1 is the selection criterion because the labels are imbalanced and each class is relevant to the task. Test metrics are reported only after selection. Confusion matrices use true labels as rows and predicted labels as columns. Interpretation considers errors between visually similar categories and the small trash test support of 18 images, rather than interpreting an aggregate score as uniform performance.

Uncertainty is summarized using 2,000 paired bootstrap resamples of test groups, retaining related images together. The same resampled groups are used for paired model comparisons. The reported 95% percentile intervals are descriptive and conditional on this fitted-model set, dataset, and split. They do not capture variation from repeated training seeds or independently sampled campuses. No formal hypothesis tests or significance claims are planned for this initial experiment.

The submission retains predictions, training histories, comparison tables, confusion-matrix plots, and a preliminary training-results PDF as supporting evidence. The Week 16 Results and Discussion manuscript will interpret those artifacts separately. An independent local-photo assessment remains future work; any such evaluation should preserve the frozen model, identify distinct physical items, prevent reuse of source images, document class coverage, and keep its results separate from TrashNet test scores.

### 2.7 Research Ethics

The dataset is attributed to its original creators and classified as secondary throughout this manuscript. Publicly downloaded photographs are not presented as photographs collected by the group. The accompanying source record, pinned archive reference, checksums, and license notice support traceability. No human participants were recruited and no survey responses or private personal records were used for this initial study.

If local photographs are collected later, the team should obtain relevant site permission, avoid faces and identifying documents, remove unnecessary location metadata, and document collection dates and labeling decisions. Permission and consent must be obtained where required before collecting identifiable material. Such collection is a prospective extension, not a completed part of the present dataset.

Responsible use also requires clear technical boundaries. A material prediction does not certify cleanliness, recycling eligibility, or safe handling of a hazardous item. The application should not replace local disposal instructions or trained personnel. Scores are uncalibrated, and unfamiliar inputs can still receive an incorrect six-class prediction. The study reports failures and limitations alongside successes and does not claim a field-tested recycling intervention.

## References

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20, 273–297. https://doi.org/10.1007/BF00994018

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 1, 886–893. https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. IEEE Conference on Computer Vision and Pattern Recognition, 248–255. https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of Machine Learning Research, 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations. https://arxiv.org/abs/1711.05101

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Proença, P. F., & Simões, P. (2020). TACO: Trash annotations in context for litter detection. arXiv. https://arxiv.org/abs/2003.06975

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. IEEE Conference on Computer Vision and Pattern Recognition, 4510–4520. https://arxiv.org/abs/1801.04381

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6, Article 60. https://doi.org/10.1186/s40537-019-0197-0

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Thung, G., & Yang, M. (n.d.). TrashNet [Data set and source repository]. GitHub. Retrieved November 7, 2026, from https://github.com/garythung/trashnet
