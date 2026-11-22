# Primary set (synthetic): datasheet

**Status:** code and protocol ready. The images and results appear after the group runs [the Colab notebook](../notebooks/WasteSort_Primary_Colab.ipynb) and commits its outputs.

## What it is

The primary set has 2,521 **AI-generated** images of single waste items. It has exactly the same class counts as the usable TrashNet images: cardboard 403, glass 498, metal 409, paper 594, plastic 480 and trash 137. Equal size and class mix mean differences between the datasets come from the images, not from sample size or class balance. The group did not photograph real objects. The course instructor approved synthetic images as the study's primary dataset. The study compares this set with the secondary dataset, TrashNet.

Call it **"primary set (synthetic)"** everywhere: in the manuscript, the slides and the app. Every JPEG also contains the disclosure as a file comment.

## Why it exists

TrashNet shows single items on a plain white background. A campus app would see grass, desks, walkways, bins, mixed lighting and dirty or crushed items. The primary set adds those conditions on purpose. That lets the study measure how far TrashNet training transfers beyond its own photo style. A real photo study was not possible in the available time.

## How it was made

All settings are in [protocol_primary.json](../protocol_primary.json). They were fixed before any image was generated.

| Step | Method |
|---|---|
| Prompt plan | 10 items per class × 12 campus settings (120 item–setting groups per class), cycled in order. The condition (clean, crushed, wet…), lighting and camera angle are drawn with seed 42 + class index. Template: *"a realistic smartphone photo of {condition} {item}, {setting}, {lighting}, {angle}…"* |
| Generator | [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo), revision `71153311`, 4 steps, guidance 0, 512 × 512, one fixed seed per candidate |
| Frame | Centre-cropped to 4:3 and resized to 512 × 384, the TrashNet resized-archive frame, so both datasets are distorted the same way when resized to 160 × 160 |
| Candidates | 1.8 × each class target (4,541 total), so quality control can reject freely. The plan is prefix-stable: raising the multiplier only adds new candidates |
| Quality control | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) zero-shot scores against six class descriptions. The rule is in `accept_candidate` in [primary_set.py](../wastesort/primary_set.py) |
| Deduplication | Exact pixel duplicates and perceptual-hash near-duplicates (Hamming ≤ 6) within the set and against every TrashNet image |
| Class counts | Round-robin over items until each class reaches its TrashNet count. Below 95% of a target, the pipeline stops |
| Split | Groups are class + item + setting. `StratifiedGroupKFold(20, seed 42)` with the same 70/15/15 fold mapping as TrashNet. Groups never cross splits |

## Files

- `data/primary_set/images/<class>/*.jpg`: the accepted images (committed)
- `data/primary_set/splits.csv`: manifest with prompt, seed, item, setting, CLIP score, hashes, group and split
- `data/primary_set/candidates.csv`: every planned candidate and its CLIP scores
- `data/primary_set/provenance.json`: generator, checker, library versions, GPU and checksums
- `reports/primary_set/audit.json`, `excluded.csv` and `split_counts.csv`: QC counts and every rejection reason
- `reports/primary_set/contact_sheet.html`: every accepted image with its label, for human review

## Comparison with the secondary data

`python -m wastesort.dataset_compare` writes `reports/primary_set/dataset_comparison.json` and charts. It measures:

- class balance
- brightness, contrast, saturation, colourfulness, edge density and **white-background fraction**
- a 2-D PCA of ImageNet MobileNetV2 features
- **domain separability**: the cross-validated accuracy of a simple classifier telling the two datasets apart. 50% means the datasets can't be told apart.

## Experiments

`python -m wastesort.primary_experiments` runs:

1. **Domain shift.** The five frozen TrashNet checkpoints are tested on the primary set without any change.
2. **Fine-tuning.** MobileNetV2 is trained on TrashNet train + primary train with the original `protocol.json` settings. Its epoch is chosen on combined validation, then it is tested once on both test splits.

The original deployment model and the published TrashNet results do not change. The fine-tuned model is an optional second choice in the app.

## Limitations (state these in the manuscript)

- **Not real photographs.** Results show transfer to generated scenes, not accuracy on real campus waste.
- **Generator bias.** SDXL-Turbo has its own visual style and can draw objects that don't exist or have the wrong material. Labels come from the prompt, not from inspecting a physical item.
- **Selection bias from CLIP.** QC keeps images that CLIP recognises, so accepted images may be easier than average. Report the rejection counts from `audit.json` with every result.
- **Circularity.** A pretrained vision model filters the data, and pretrained models are then evaluated on it.
- **Same trash shortage.** Mirroring TrashNet keeps only 137 trash images, about 18 in the test split, so trash recall stays uncertain in both datasets.
- **One seed and one generator.** Diffusion output can differ slightly between GPUs and library versions. The saved images and their hashes are the record.
- **License.** SDXL-Turbo is under the Stability AI non-commercial research community license. The primary set is for this non-commercial course study only.

## Human review still needed

Open the contact sheet and check every label. Record unclear or wrong images in a separate review note with dates, as in `docs/DUPLICATE_REVIEW.md`. Do not silently drop or relabel images after seeing model results.
