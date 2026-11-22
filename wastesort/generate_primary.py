"""Generate the primary set (synthetic) candidates on a GPU and score them with CLIP.

Run in Colab: python -m wastesort.generate_primary
Interrupted runs resume: existing candidate files are kept and not regenerated.
"""
import argparse
import importlib.metadata
import platform
import time

import pandas as pd
from PIL import Image

from .common import CLASSES, ROOT, save_json
from .primary_set import DATA, DISCLOSURE, build_prompts, load_protocol


def crop_to_frame(image, size):
    """Centre-crop to the saved aspect ratio, then resize to the TrashNet frame."""
    width, height = size
    target_ratio = width / height
    if image.width / image.height > target_ratio:
        new_width = round(image.height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    else:
        new_height = round(image.width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))
    return image.resize(size, Image.Resampling.LANCZOS)


def generate(rows, generator_settings, batch_size):
    import torch
    from diffusers import AutoPipelineForText2Image
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required. In Colab choose Runtime -> Change runtime type -> GPU.")
    pipe = AutoPipelineForText2Image.from_pretrained(
        generator_settings["model"], revision=generator_settings["revision"],
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    todo = [r for r in rows if not (ROOT / r["candidate_path"]).exists()]
    print(f"{len(rows) - len(todo)} candidates already exist; generating {len(todo)}", flush=True)
    started = time.perf_counter()
    width, height = generator_settings["generate_size"]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        seeds = [torch.Generator("cuda").manual_seed(int(r["seed"])) for r in batch]
        images = pipe(prompt=[r["prompt"] for r in batch], generator=seeds,
                      num_inference_steps=generator_settings["steps"],
                      guidance_scale=generator_settings["guidance_scale"],
                      width=width, height=height).images
        for row, image in zip(batch, images):
            target = ROOT / row["candidate_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            framed = crop_to_frame(image.convert("RGB"), tuple(generator_settings["saved_size"]))
            # The disclosure travels with each file; JPEG comments survive copying.
            framed.save(target, "JPEG", quality=generator_settings["jpeg_quality"],
                        comment=DISCLOSURE.encode("utf-8"))
        done = start + len(batch)
        if done % (batch_size * 10) == 0 or done == len(todo):
            print(f"generated {done}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)
    del pipe
    torch.cuda.empty_cache()


def clip_scores(rows, qc):
    """Zero-shot probabilities over the six class descriptions for every existing candidate."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(qc["checker"], revision=qc["checker_revision"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(qc["checker"], revision=qc["checker_revision"])
    texts = [qc["text_template"].format(description=qc["class_descriptions"][c]) for c in CLASSES]
    present = [r for r in rows if (ROOT / r["candidate_path"]).exists()]
    for start in range(0, len(present), 32):
        batch = present[start:start + 32]
        images = [Image.open(ROOT / r["candidate_path"]).convert("RGB") for r in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            probabilities = model(**inputs).logits_per_image.softmax(-1).cpu().numpy()
        for row, values in zip(batch, probabilities):
            for name, value in zip(CLASSES, values):
                row[f"clip_{name}"] = float(value)
    for row in rows:
        for name in CLASSES:
            row.setdefault(f"clip_{name}", float("nan"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate only the first N candidates per class (quick trial run).")
    args = parser.parse_args()
    protocol = load_protocol()
    rows = build_prompts(protocol)
    if args.limit:
        rows = [r for r in rows if int(r["candidate_id"].rsplit("_", 1)[1]) < args.limit]
    generate(rows, protocol["generator"], args.batch_size)
    rows = clip_scores(rows, protocol["quality_control"])
    DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "candidates.csv", index=False, lineterminator="\n")
    import torch
    save_json(DATA / "generation_environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "diffusers", "transformers", "accelerate", "pillow"]},
        "note": "Diffusion output can differ slightly across GPUs and library builds even with fixed seeds; "
                "the saved images and their SHA-256 values are the record.",
    })
    print(f"Saved {len(rows)} candidate rows to {DATA / 'candidates.csv'}", flush=True)


if __name__ == "__main__":
    main()
