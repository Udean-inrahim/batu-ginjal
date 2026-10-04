"""Prepare paired, human-annotated YOLO labels for a coronal Stone baseline."""
from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "ct_annotated"
OUT = ROOT / "dataset" / "coronal_stone"
SEED = 76021


def paired_images(split):
    images = SOURCE / "images" / split
    labels = SOURCE / "labels" / split
    pairs = []
    for image in sorted(images.glob("*.jpg")):
        label = labels / f"{image.stem}.txt"
        if label.exists() and label.read_text(encoding="utf-8").strip():
            pairs.append((image, label))
    return pairs


def copy_pairs(pairs, split):
    image_dir, label_dir = OUT / "images" / split, OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        shutil.copyfile(image, image_dir / image.name)
        shutil.copyfile(label, label_dir / label.name)


def main():
    source_train = paired_images("valid")
    source_test = paired_images("test")
    if len(source_train) < 20 or len(source_test) < 20:
        raise SystemExit(f"Pasangan berlabel terlalu sedikit: train={len(source_train)}, test={len(source_test)}")

    rng = random.Random(SEED)
    rng.shuffle(source_train)
    validation_size = max(12, round(len(source_train) * 0.2))
    train_pairs = source_train[validation_size:]
    val_pairs = source_train[:validation_size]

    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            folder = OUT / kind / split
            if folder.exists():
                for old in folder.glob("*"):
                    if old.is_file():
                        old.unlink()

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(source_test, "test")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: Stone\n", encoding="utf-8")
    (OUT / "source.json").write_text(json.dumps({
        "source": "dataset/ct_annotated",
        "train_and_val_source_split": "valid",
        "external_test_source_split": "test",
        "counts": {"train": len(train_pairs), "val": len(val_pairs), "test": len(source_test)},
        "class": "Stone",
        "orientation": "coronal CT images as supplied by source"
    }, indent=2), encoding="utf-8")
    print(json.dumps({"train": len(train_pairs), "val": len(val_pairs), "test": len(source_test),
                      "yaml": str(OUT / 'data.yaml')}, indent=2))


if __name__ == "__main__":
    main()
