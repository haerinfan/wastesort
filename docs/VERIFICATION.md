# Verification record

Executed locally on October 20, 2026 with Python 3.12.14, Windows 11 and an NVIDIA GeForce RTX 2080. Package and CUDA versions are in [environment.json](../reports/environment.json).

## Completed checks

- Downloaded the real resized TrashNet archive, recorded its checksum, decoded all images and recorded exclusions.
- Regenerated the frozen split identically; checked image/group separation, all six classes in each partition and manifest/protocol hashes.
- Executed classical training, neural training, validation selection, final held-out evaluation, bootstrap summaries and the separate numerical regression exercise.
- Ran eight local automated tests successfully, covering upload validation, features, grouping, split integrity, preprocessing, deployment selection and Streamlit behavior.
- Tested the live app in Edge: uploaded `plastic5.jpg`, ran prediction and verified the displayed Plastic result and 86.5% uncalibrated score. Checked the research-results screen. This was an interface check, not an additional accuracy estimate.
- Validated the Colab notebook structure. Its pipeline modules were executed locally.
- Rendered the revised proposal through Microsoft Word and checked the pages. Rendered and checked the revised short-bond status sheet.
- Exported the seven-slide Monday deck through Microsoft PowerPoint; checked all seven slide images and both speaking-guide pages. The deck contains slide-by-slide speaker notes.

An initial neural run stopped because CUDA's deterministic backward operation did not support adaptive average pooling. The implementation uses a fixed 5 × 5 average pool on the 20 × 20 feature map, mathematically equivalent to the planned 4 × 4 adaptive output for the fixed 160 × 160 input. Neural training was restarted before final test evaluation. This implementation detail is documented here without rewriting the frozen protocol.

## What these checks do not establish

The notebook has not been executed inside Google Colab in this session. A full clean-environment training reproduction by a group member is pending. GPU and platform changes can affect exact numerical reproduction.

No complete human image/label audit, independent campus-photo validation, campus disposal-rule verification or instructor acceptance has been completed. Revised proposal status fields retain planning/source context; executed technical results are recorded separately. Monday materials are prepared, but presentation delivery and individual mastery remain to be demonstrated.

See [group checklist](GROUP_CHECKLIST.md) and [contribution log](../CONTRIBUTIONS.md) for follow-up work.
