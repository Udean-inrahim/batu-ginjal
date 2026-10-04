"""Download Mask R-CNN pseudo-label CT set and convert polygons to YOLO labels."""
from pathlib import Path
import json
import random
import re
import shutil
import sys

from datasets import load_dataset
from PIL import Image
import numpy as np

REPO = "ryfkn/CT-Kidney-Dataset-Stone-Predicted-Mask-RCNN-v2"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "yolo_stone"
MAX_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def main():
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            target = OUT / kind / split
            if target.exists():
                for old_file in target.iterdir():
                    if old_file.is_file():
                        old_file.unlink()
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    api = HfApi()
    shards = sorted(name for name in api.list_repo_files(REPO, repo_type="dataset")
                    if name.startswith("data/") and name.endswith(".parquet"))
    if not shards:
        raise RuntimeError("Tidak menemukan data Parquet dataset.")
    urls = [f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}" for name in shards]
    dataset = load_dataset("parquet", data_files=urls, split="train", streaming=True)
    rows = list(dataset.take(MAX_IMAGES)) if MAX_IMAGES > 0 else list(dataset)
    random.Random(2026).shuffle(rows)
    n = len(rows)
    train_end, val_end = int(n * .70), int(n * .90)
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for index, row in enumerate(rows, 1):
        try:
            image = row["ct_image"].convert("RGB")
            mask = row["instance_mask_union"].convert("L")
            width, height = image.size
            image_name = re.sub(r"[^A-Za-z0-9_.-]", "_", row["id"]) + ".jpg"
            split = "train" if index <= train_end else "val" if index <= val_end else "test"
            image_target = OUT / "images" / split / image_name
            image.save(image_target, quality=95)
            label_target = OUT / "labels" / split / f"{Path(image_name).stem}.txt"
            rows = []
            binary = np.asarray(mask) > 0
            ys, xs = np.where(binary)
            if len(xs):
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if x2 > x1 and y2 > y1:
                    xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                    bw, bh = (x2 - x1) / width, (y2 - y1) / height
                    rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            label_target.write_text("\n".join(rows), encoding="utf-8")
            counts[split] += 1
            if index % 20 == 0 or index == n:
                print(f"Diproses {index}/{n}, berhasil {sum(counts.values())}, lewati {skipped}", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"Lewati {row.get('id', index)}: {exc}", flush=True)

    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Stone\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset": str(OUT), "counts": counts, "skipped": skipped,
                      "pseudo_labels": True, "yaml": str(data_yaml)}, indent=2))


if __name__ == "__main__":
    main()
