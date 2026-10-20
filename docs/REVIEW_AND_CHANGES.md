# Review and implemented decisions

The supplied proposal is feasible as a controlled, single-image research prototype. The supplied weekly sheet is an introductory learning plan. Their model lists and deadlines needed explicit separation.

| Finding | Change |
|---|---|
| Proposal has four final approaches; weekly sheet has KNN and Decision Tree | Preserve both: four primary research methods, two introductory validation-only exercises. |
| Linear Regression is unrelated to material labels | Keep it as a separate numerical learning exercise using scikit-learn's bundled diabetes benchmark, without medical interpretation. |
| October 24 demo conflicts with the later prototype milestone | Define October 24 as the initial baseline demonstration. Keep the full research completion schedule. |
| Future dates appeared under actual start/finish headings | Separate planned dates from actual dates. Preserve supplied actual starts for the first two activities; leave future actuals pending. |
| Model-selection metric was not consistently tied to validation | Select the deployment using validation macro F1 before exposing test results. |
| Statistical tests were unspecified | Replace formal null-hypothesis claims with descriptive comparisons and 2,000 paired group-bootstrap intervals for the initial single-seed study. Instructor review remains pending. |
| Weighting hypothesis had no bounded experiment | Add one weighted-versus-unweighted small-CNN experiment with matched seed and training settings. |
| TensorFlow/Keras was proposed | Use PyTorch/torchvision for native Windows GPU execution and Colab portability; retain the same model families. |
| Duplicate leakage could inflate results | Audit exact pixels, exclude all copies with conflicting labels, group perceptual-hash near-duplicate candidates before splitting and publish exclusions/manifests. |
| Original and resized upstream datasets coexist | Pin and download only the official resized archive. |
| Campus bin rules are unconfirmed | Display a pending-verification message and keep material prediction separate from disposal decisions. |
| AI disclosure covered mostly drafting | Expand disclosure to implementation and experiment execution; require human verification and truthful contribution records. |
| Numbering continued across unrelated sections | Restart the five independent numbered lists. |
| Ruiz paper had incorrect book title and author form | Correct to the publisher's Ruiz et al. citation in *From Bioinspired Systems and Biomedical Applications to Machine Learning*. |

## Sources checked

- TrashNet class counts, image conditions, attribution and source: https://github.com/garythung/trashnet
- Dataset hosting, original/resized archive separation and MIT metadata: https://huggingface.co/datasets/garythung/trashnet/tree/main
- Ruiz et al. publisher citation: https://link.springer.com/chapter/10.1007/978-3-030-19651-6_41
- MobileNetV2 implementation and weights: https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v2.html
- PyTorch compatible Windows builds: https://pytorch.org/get-started/previous-versions/
- Group-preserving stratification: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

The entire literature review has not been independently verified against full texts. The supplied proposal's topic-approval claim is retained as a source statement, not independently verified approval. The revised protocol is a working draft for instructor review.

## Revisions and progress are different

The revised documents describe the working plan. They do not certify individual attendance, contributions, instructor acceptance or campus-rule verification. The repository's measured results document automated execution, while group members must review the scientific interpretation.

The source documents in Downloads were not modified. Revised copies are in `documents/`.
