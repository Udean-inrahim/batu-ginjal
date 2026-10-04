"""Combine patient-split kidney masks and CT stone pseudo-labels for research only."""
from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
KIDNEY = ROOT / "dataset" / "kits23_kidney"
STONE = ROOT / "dataset" / "yolo_stone"
CT_ANNOTATED = ROOT / "dataset" / "ct_annotated"
OUT = ROOT / "dataset" / "kidney_stone_joint"
RNG = random.Random(5412)
TRAIN_CAPS = {"kidney": 300, "pseudo_stone": 180}


def main():
    counts = {s: {"Kidney": 0, "Stone": 0} for s in ("train", "val", "test")}
    for split in counts:
        images = OUT / "images" / split
        labels = OUT / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for file in images.iterdir():
            if file.is_file(): file.unlink()
        for file in labels.iterdir():
            if file.is_file(): file.unlink()

    # Kidney masks are converted to class 0. The split is by patient case.
    for split in counts:
        images = OUT / "images" / split
        labels = OUT / "labels" / split
        kidney_images = sorted((KIDNEY / "images" / split).glob("*.jpg"))
        if split == "train" and len(kidney_images) > TRAIN_CAPS["kidney"]:
            kidney_images = sorted(RNG.sample(kidney_images, TRAIN_CAPS["kidney"]))
        for source_image in kidney_images:
            source_label = KIDNEY / "labels" / split / f"{source_image.stem}.txt"
            if not source_label.exists():
                continue
            shutil.copyfile(source_image, images / source_image.name)
            rows = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) == 5:
                    values[0] = "0"
                    rows.append(" ".join(values))
            (labels / f"{source_image.stem}.txt").write_text("\n".join(rows), encoding="utf-8")
            counts[split]["Kidney"] += 1

        # Stone labels are predicted Mask R-CNN masks, not expert ground truth.
        pseudo_images = sorted((STONE / "images" / split).glob("*.jpg"))
        if split == "train" and len(pseudo_images) > TRAIN_CAPS["pseudo_stone"]:
            pseudo_images = sorted(RNG.sample(pseudo_images, TRAIN_CAPS["pseudo_stone"]))
        for source_image in pseudo_images:
            source_label = STONE / "labels" / split / f"{source_image.stem}.txt"
            if not source_label.exists():
                continue
            target_stem = f"pseudo_{source_image.stem}"
            shutil.copyfile(source_image, images / f"{target_stem}.jpg")
            rows = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) == 5:
                    values[0] = "1"
                    rows.append(" ".join(values))
            (labels / f"{target_stem}.txt").write_text("\n".join(rows), encoding="utf-8")
            counts[split]["Stone"] += 1

    # This paired image/YOLO split contains CT images in a different plane than
    # KiTS23. Use its validation partition for supervised stone fine-tuning while
    # keeping its test partition untouched as a domain-specific holdout.
    coronal_counts = {"train": 0, "test": 0}
    for source_split, target_split in (("valid", "train"), ("test", "test")):
        source_images = CT_ANNOTATED / "images" / source_split
        source_labels = CT_ANNOTATED / "labels" / source_split
        if not source_images.exists():
            continue
        for source_image in source_images.glob("*.jpg"):
            source_label = source_labels / f"{source_image.stem}.txt"
            if not source_label.exists():
                continue
            target_stem = f"coronal_{source_image.stem}"
            shutil.copyfile(source_image, OUT / "images" / target_split / f"{target_stem}.jpg")
            rows = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) == 5:
                    values[0] = "1"
                    rows.append(" ".join(values))
            (OUT / "labels" / target_split / f"{target_stem}.txt").write_text("\n".join(rows), encoding="utf-8")
            counts[target_split]["Stone"] += 1
            coronal_counts[target_split] += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: Kidney\n  1: Stone\n",
        encoding="utf-8",
    )
    (OUT / "provenance.json").write_text(json.dumps({
        "kidney_source": "KiTS23 sliced CT kidney masks, grouped by case",
        "stone_source": "Mask R-CNN pseudo masks plus paired coronal CT YOLO labels; mixed annotation provenance",
        "coronal_supervised_images": coronal_counts,
        "counts": counts,
    }, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
