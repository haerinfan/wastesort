# Initial measured results

Executed October 20, 2026. These are real training and held-out evaluation results, not illustrative scores. This is one seed (42) and one grouped split, with human image review still pending.

## Data and selection

TrashNet supplied 2,527 images. The automated audit excluded all six files in three identical-pixel pairs with conflicting labels. The remaining 2,521 images form 2,476 perceptual-hash candidate groups: 1,774 training, 371 validation and 376 test images. Candidate groups do not cross partitions. Similarity grouping reduces detected leakage; it does not prove that every repeated physical object has been found.

MobileNetV2 had the highest validation macro F1 among the four primary methods. Its deployment record was frozen at 09:01:18 UTC, before final test evaluation at 09:02:19 UTC. The weighting ablation and introductory exercises were excluded from deployment selection.

## Comparison

All values below are percentages, except where stated.

| Method | Validation macro F1 | Test accuracy | Test macro F1 | Test trash recall |
|---|---:|---:|---:|---:|
| HOG + SVM | 60.62 | 65.96 | 63.52 | 38.89 |
| HOG + Random Forest | 55.94 | 63.56 | 60.22 | 33.33 |
| Small CNN | 62.07 | 59.57 | 57.68 | 55.56 |
| **MobileNetV2** | **84.74** | **84.57** | **81.22** | **55.56** |
| Small CNN, weighted loss (ablation) | 60.32 | 58.78 | 58.84 | 72.22 |

Accuracy counts correct predictions overall. Macro F1 averages the six class F1 scores equally, so larger classes do not dominate the average.

MobileNetV2 classified 318 of 376 held-out images correctly. It recognized only 10 of 18 trash images. Thus the overall score hides an important minority-class weakness. Its best checkpoint was epoch 8; training stopped after epoch 13.

The weighted CNN recognized 13 of 18 trash images, compared with 10 for the unweighted CNN, but its overall accuracy fell. This single run does not establish that weighting reliably improves performance.

## Uncertainty

The 2,000 paired group-bootstrap resamples give a descriptive 95% interval of 76.25–85.60% for MobileNetV2 macro F1. The MobileNetV2-minus-CNN macro-F1 difference interval is +17.25 to +30.21 percentage points. The weighted-minus-unweighted CNN intervals are −3.86 to +6.12 points for macro F1 and −7.69 to +40.00 points for trash recall.

These intervals are conditional on the fitted models and this test sample. They do not measure variability over training seeds, establish campus accuracy, or constitute the originally unspecified formal hypothesis tests.

## Efficiency and exercises

MobileNetV2's saved checkpoint is 8.73 MiB. Mean CPU inference was 11.30 ms over 30 single-image calls after three warmups, using four Torch threads. Preprocessing was included; file decoding and browser overhead were excluded. This is a local measurement, not a service-speed guarantee.

The recorded training durations differ in scope: classical durations sum estimator fitting across candidate settings and report shared HOG extraction separately; neural durations include the training/validation loop but exclude initial data loading and pretrained-weight downloads. They are not a controlled hardware-efficiency comparison.

KNN and Decision Tree were evaluated on validation data for introductory learning only. Linear Regression used the separate bundled numerical diabetes benchmark: 353 training rows, 89 test rows, MSE 2,900.19 and R² 0.4526. This numerical exercise is not used in the waste classifier and has no clinical interpretation.

## Evidence and limits

- [Machine-readable comparison](../reports/comparison.csv)
- [Full metrics and class supports](../reports/test_evaluation.json)
- [Saved test predictions](../reports/test_predictions.csv)
- [Bootstrap intervals](../reports/uncertainty.json)
- [Dataset audit](../reports/data_audit.json)
- [Error examples](../reports/error_examples.png)
- [Training example grid](../reports/training_examples.png)

The dataset mostly shows single items against simple backgrounds. No independent campus-photo evaluation has been completed. Unrelated objects, multiple items, contamination and hazardous waste remain outside scope. No model was changed in response to the final test results.

## Primary dataset

Executed November 23, 2026 on one NVIDIA RTX 2080 (`--low-vram`), Python 3.13 with the pinned packages. Reported separately from the TrashNet results above, which are unchanged.

**Generated images.** 4,541 candidates; 1,428 failed the CLIP rule (1,298 wrong top class, 105 ambiguous, 25 low confidence), 1 near-duplicate, 595 surplus. 2,517 kept: every class at its TrashNet target except plastic, 476 of 480 (the protocol accepts at least 95%). Grouped splits: 1,765 training, 361 validation, 391 test (640 groups). Source: `reports/primary_set/audit.json`.

**Dataset comparison** (`dataset_comparison.json`). Mean brightness 0.64 TrashNet vs 0.44 generated; saturation 0.18 vs 0.39; edge density 0.13 vs 0.21; near-white fraction 0.085 vs 0.018. A domain classifier separates the datasets with 98.97% accuracy (50% = indistinguishable).

**Frozen TrashNet models on all generated images** (`domain_shift.csv`, macro F1):

| Model | TrashNet test | Generated | Generated trash recall |
|---|---:|---:|---:|
| HOG + SVM | 63.52% | 11.37% | 0.73% |
| HOG + Random Forest | 60.22% | 10.02% | 0.00% |
| Small CNN | 57.68% | 22.31% | 4.38% |
| MobileNetV2 | 81.22% | 59.73% | 20.44% |
| Small CNN, weighted | 58.84% | 20.04% | 8.03% |

**MobileNetV2 fine-tuned on TrashNet + generated images** (`finetune_comparison.csv`; epoch 12 chosen on combined validation macro F1 0.8447 before testing):

| Trained on | Test set | Accuracy | Macro F1 | Trash recall |
|---|---|---:|---:|---:|
| TrashNet | TrashNet test (376) | 84.57% | 81.22% | 55.56% |
| TrashNet | Generated test (391) | 63.43% | 58.50% | 22.22% |
| TrashNet + generated | TrashNet test (376) | 84.84% | 82.45% | 66.67% |
| TrashNet + generated | Generated test (391) | 90.54% | 87.66% | 77.78% |

**Real photographs** (`real_photos.csv`, 9 scored photos, three classes): SVM 1/9, Random Forest 0/9, small CNN 1/9, weighted CNN 2/9, MobileNetV2 3/9, fine-tuned MobileNetV2 5/9. Nine photos show failure cases; they do not estimate accuracy.

Generated-image results measure transfer to AI-generated scenes, not accuracy on real campus waste.
