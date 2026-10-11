"""Export human-reviewed axial candidates into a patient-disjoint YOLO dataset."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Stone"]


def split_for_patient(patient):
    bucket = int(hashlib.sha256(patient.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def valid_box(bbox, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="runs/axial_autolabel/candidates_stone.json")
    parser.add_argument("--out", default="dataset/axial_reviewed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    queue_path = ROOT / args.queue
    if not queue_path.is_file():
        raise SystemExit(f"Antrean tidak ditemukan: {queue_path}")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    source = Path(queue["source"])

    out_root = ROOT / args.out
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output sudah ada, pakai --overwrite: {out_root}")
        shutil.rmtree(out_root)

    counts = {"train": 0, "val": 0, "test": 0}
    object_counts = {"train": 0, "val": 0, "test": 0}
    positives = {"train": 0, "val": 0, "test": 0}
    negatives = {"train": 0, "val": 0, "test": 0}
    reviewed = 0
    skipped = 0

    for record in queue["records"]:
        if not record.get("reviewed"):
            continue
        reviewed += 1
        width, height = record["width"], record["height"]
        split = split_for_patient(record["patient"])

        boxes = []
        for item in record.get("accepted", []):
            box = valid_box(item["bbox"], width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            boxes.append((0, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                          (x2 - x1) / width, (y2 - y1) / height))

        image_src = source / record["image"]
        if not image_src.is_file():
            skipped += 1
            continue

        image_out = out_root / "images" / split
        label_out = out_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, image_out / record["image"])

        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
        (label_out / f"{Path(record['image']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        counts[split] += 1
        object_counts[split] += len(boxes)
        if boxes:
            positives[split] += 1
        else:
            negatives[split] += 1

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames:\n{names}\n", encoding="utf-8")
    provenance = {
        "source_queue": str(queue_path.resolve()),
        "reviewed_images": reviewed,
        "skipped_missing_image": skipped,
        "image_counts": counts,
        "positive_images": positives,
        "negative_images": negatives,
        "object_counts": object_counts,
        "classes": CLASSES,
        "warning": "Labels originate from model proposals confirmed by a human reviewer; not clinically validated.",
    }
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
