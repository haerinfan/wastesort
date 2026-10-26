# Goal Diggers - Monday to Sunday

Updated October 26, 2026. This is a joint, AI-assisted working record requested through the shared chat. Day names identify learning activities, not a claim that every task was completed on its originally planned date. The supplied WIG schedule covers Monday through Saturday; Sunday is an added verification and handoff activity.

## Monday - Explain the problem

AI is the broad field of building systems that perform tasks associated with intelligence. Machine learning is an approach that learns patterns from examples. In supervised learning, examples include target labels. Unsupervised learning searches for structure without those labels; reinforcement learning learns actions through rewards. WasteSort is supervised multiclass image classification: the input is an item photo and the output is one of six material classes. It does not learn a numerical waste score or automatically establish the correct campus bin.

Evidence: the Monday PowerPoint, PDF, speaking guide and brief in `presentation/`.

Practice answer: A labeled plastic-bottle image is a supervised example. The material label is the target. An unseen photo is processed using the same transformations before prediction.

## Tuesday - Prepare and inspect the data

TrashNet contains 2,527 source images. The complete inventory records paths, labels, dimensions, format and inclusion status. Six files in three identical-pixel pairs have conflicting labels; both copies of each pair are excluded. The usable dataset has 2,521 images. Candidate groups stay within one partition: 1,774 training, 371 validation and 376 test images, seed 42.

All 45 flagged similarity pairs now have dated AI-assisted visual observations: 28 likely same-item, 11 visibly distinct, 6 uncertain. The six excluded-file worksheet entries also record retention of the original exclusions. Uncertain is a completed review decision, not a guessed label. No published labels or partitions were changed after viewing test results. See `docs/DUPLICATE_REVIEW.md`.

HOG models use 64 x 64 grayscale inputs and 1,764 features. Neural models use 160 x 160 RGB inputs; MobileNetV2 additionally uses pretrained normalization. Random augmentation applies only to training. Label codes 0-5 identify categories, not quantities.

Evidence: `reports/dataset_inventory.csv`, `reports/manual_review_worksheet.csv`, the Tuesday presentation and preparation notebook.

## Wednesday - KNN and Decision Tree

KNN compares a new HOG feature vector with nearby training vectors; this pipeline standardizes features before applying KNN. The Decision Tree uses feature-based branching rules, with depth limited to 12 and balanced class weights. Both were trained on training data. Validation predictions are saved without retraining or changing model choices.

KNN validation accuracy is 60.65% and macro F1 is 57.03%. Decision Tree validation accuracy is 36.93% and macro F1 is 35.04%. These are validation results, not final test scores. They must not be ranked against other models' test scores as if the data were the same.

Evidence: `reports/intro_validation_predictions.csv` contains 742 rows (371 per model), with path, actual label, predicted label and correctness. `reports/intro_sample_predictions.csv` contains two validation examples per true class per model, selected by manifest order without filtering for correct predictions. Both validation confusion matrices and metrics are saved.

Reproduce: `python -m wastesort.intro_predictions` after the dataset is prepared. Read a sample row aloud, identify its true and predicted materials, and explain why one example does not measure overall accuracy.

## Thursday - Numerical regression

Linear Regression predicts a continuous value using a weighted combination of numerical inputs. This separate exercise uses scikit-learn's bundled diabetes benchmark with 353 training rows, 89 test rows and seed 42. It is educational and is not used for waste labels or clinical decisions.

The mean squared error is 2,900.19: the average squared difference between a prediction and its true target, in squared target units. Squaring gives large errors more weight. R-squared is 0.453, meaning this model explains about 45.3% of target variation relative to a constant test-mean reference. It is not 45.3% classification accuracy and does not mean every prediction is close.

Evidence: `reports/regression_predictions.csv` holds all 89 actual and predicted values; `reports/regression_exercise.json` records the metrics. Reproduce with `python -m wastesort.exercises` in a fresh run directory.

## Friday - Evaluate and explain

Accuracy is the fraction of predictions that match their labels. Precision asks how many predictions of a class are correct. Recall asks how many actual members of a class were found. F1 balances precision and recall; macro F1 averages class F1 scores equally, so large classes do not dominate it. A confusion matrix shows actual labels by row and predictions by column; the diagonal contains correct predictions.

The selected MobileNetV2 obtained 318/376 correct test predictions: 84.57% accuracy and 81.22% macro F1. It found only 10/18 trash images (55.56% recall), which is an important weakness. The primary models were selected and compared using the documented validation/test protocol. The weighted CNN is a separate ablation; it was not substituted after inspecting test results. Regression MSE is reported separately because it measures a different problem.

Evidence: `docs/RESULTS.md`, `reports/comparison.csv`, per-model confusion matrices and `reports/test_predictions.csv`. One split and one training seed support a descriptive benchmark, not a formal hypothesis rejection or a campus accuracy claim.

## Saturday - Demonstrate the workflow

Use the six-slide Saturday deck in `presentation/day6/`. Start WasteSort, open its local browser page, upload one clear JPEG/PNG, select Identify material, read the result and download the JSON. Show the research-results table and explain the trash-class weakness. If a dataset photo is used, describe it as a demonstration, not an independent campus test.

The full Colab notebook prepares a fresh experiment, trains the models, performs fixed-protocol evaluation, runs the regression exercise and exports evidence. Its output records and environment are separate from the original published Windows run. Do not silently replace the original model or scores with a more favorable rerun.

Presentation script: We are Goal Diggers. We prepared labeled images, kept candidate duplicate groups together, trained and compared models, and built a single-photo prototype. MobileNetV2 led our published benchmark but still missed eight of eighteen trash images. Its output helps explore material appearance; campus disposal rules and a separate campus-photo evaluation remain outside this completed benchmark.

## Sunday - Verify and hand off

Sunday was not specified in the original Monday-Saturday WIG. We add it to reconcile the gap report, review recorded evidence, verify tests and model checksums, package the full project, and publish the shared repository. Use `python -m pytest -q` and `python -m wastesort.doctor`. Check that every row of the weekly status sheet links to an actual output and that completion dates reflect evidence, not planned dates.

Defense practice prompts with model answers:

- Why group similar images? To reduce detected leakage between partitions; grouping does not guarantee every duplicate was found.
- Why use validation? To choose settings and the deployment model before final test evaluation.
- Does 84.57% mean campus accuracy? No; it describes the held-out TrashNet images in the recorded split.
- Why is trash recall important? Missing 8 of 18 actual trash examples can be hidden by stronger performance on larger classes.
- Can we reject H01-H03? Not from descriptive bootstrap intervals; the revised descriptive scope needs instructor acceptance.
- Why PyTorch instead of Keras? The implemented study uses PyTorch/torchvision; the method and reproducibility records must state that change.

These are practice materials, not evidence that an oral defense has taken place. Joint participation is reported by the user; this record does not invent individual speaking times, private work or instructor approval.
