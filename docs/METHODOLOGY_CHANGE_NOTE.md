# Methodology changes for instructor review

Prepared October 26, 2026 by Goal Diggers with AI assistance. This note is ready to submit; no instructor response or approval is asserted and no email has been sent by the assistant.

## Requested decisions

| Proposal | Implemented study | Reason and consequence |
|---|---|---|
| TensorFlow/Keras | PyTorch 2.7.1 and torchvision 0.22.1 | The actual runnable implementation and saved checkpoints use these packages. The methodology must describe the tools used. Framework substitution does not establish equivalence of results. |
| Stratified 70/15/15, seed 42 | Approximately 70/15/15 with candidate duplicate groups preserved, seed 42 | Actual counts are 1,774/371/376. Grouping reduces detected duplicate leakage; it changes the exact sampling procedure and must be disclosed. |
| H01-H03 with formal statistical testing | Descriptive model comparison and conditional bootstrap intervals | A single fitted run and split are not sufficient to claim training-seed generalization. No null hypothesis is accepted or rejected from these intervals. |

Please confirm whether this revised exploratory scope is acceptable. If formal testing or the original framework is required, the group will define a separate, instructor-approved experiment before running it, rather than modifying the already exposed test study and calling it untouched.

## Ready-to-send message

Subject: WasteSort methodology changes for review - BCS34 Goal Diggers

Good day, Professor. Our WasteSort prototype and initial experiments are implemented. We would like your approval for three documented changes: PyTorch/torchvision instead of TensorFlow/Keras; a grouped, approximately 70/15/15 split with seed 42 to keep candidate duplicate images together; and descriptive comparisons with bootstrap intervals instead of claiming formal rejection of H01-H03. The saved split has 1,774 training, 371 validation and 376 test images. We have updated the method description to match the implementation and retained the original experiment record. Please let us know whether this scope is acceptable or whether a separately planned confirmatory experiment is required. Thank you. - Goal Diggers, BCS34

## Decision record

Status: Prepared for instructor review. Approval, date of reply and requested revisions remain unrecorded because no response has been supplied. This is an external decision; completing the implementation cannot substitute for it.
