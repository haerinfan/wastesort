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
