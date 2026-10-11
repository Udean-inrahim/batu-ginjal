"""Auto-propose kidney-stone bounding boxes for the axial image-level dataset.

The axial dataset only provides Stone/Non-Stone image labels, so this script
runs the existing detector weights over full frames and overlapping tiles to
produce candidate boxes for human review. Output is a JSON review queue; it is
NOT ground truth and must be verified before training.
"""
import argparse
import json
from pathlib import Path
import re
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = (ROOT / "dataset"
           / "Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection "
             "A Resource for Deep Learning Research"
           / "Kindey Stone Dataset" / "Original")
PATIENT_PATTERN = re.compile(r"^(P\d+)_")


def patient_id(filename):
    match = PATIENT_PATTERN.match(filename)
    return match.group(1) if match else "UNKNOWN"


def tile_ranges(size, fraction=0.65):
    return [(0, min(size, int(size * fraction))),
            (max(0, int(size * (1 - fraction))), size)]


def propose_for_image(image, models, conf, imgsz):
    img_w, img_h = image.size
    scans = [(image, 0, 0)]
    if min(img_w, img_h) >= 300:
        for left, right in tile_ranges(img_w):
            for top, bottom in tile_ranges(img_h):
                scans.append((image.crop((left, top, right, bottom)), left, top))

    candidates = []
    for scan_image, offset_x, offset_y in scans:
        for model in models:
            result = model.predict(scan_image, conf=conf, imgsz=imgsz, verbose=False)[0]
            for box in result.boxes:
                label = result.names.get(int(box.cls.item()), "")
                if label.lower() != "stone":
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidates.append({
                    "bbox": [round(x1 + offset_x, 2), round(y1 + offset_y, 2),
                             round(x2 + offset_x, 2), round(y2 + offset_y, 2)],
                    "confidence": round(float(box.conf.item()), 4),
                })
    return _dedupe(candidates, img_w, img_h)


def _dedupe(candidates, img_w, img_h):
    kept = []
    for candidate in sorted(candidates, key=lambda c: c["confidence"], reverse=True):
        x1, y1, x2, y2 = candidate["bbox"]
        area_a = max(1.0, (x2 - x1) * (y2 - y1))
        cx_a, cy_a = (x1 + x2) / 2, (y1 + y2) / 2
        duplicate = False
        for existing in kept:
            a1, b1, a2, b2 = existing["bbox"]
            inter = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
            area_b = max(1.0, (a2 - a1) * (b2 - b1))
            cx_b, cy_b = (a1 + a2) / 2, (b1 + b2) / 2
            dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
            if (inter / (area_a + area_b - inter) > 0.15
                    or inter / min(area_a, area_b) > 0.35
                    or dist < max(img_w, img_h) * 0.025):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="label_class", default="Stone", choices=["Stone", "Non-Stone"])
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--patients", default="", help="comma-separated patient ids")
    parser.add_argument("--out", default="runs/axial_autolabel/candidates_stone.json")
    args = parser.parse_args()

    from ultralytics import YOLO
    models = [YOLO(str(ROOT / "model" / "best.pt"))]
    stone_path = ROOT / "model" / "stone_coronal_best.pt"
    if stone_path.is_file():
        models.append(YOLO(str(stone_path)))

    source = DATASET / args.label_class
    if not source.is_dir():
        raise SystemExit(f"Folder tidak ditemukan: {source}")

    wanted = {p.strip() for p in args.patients.split(",") if p.strip()}
    images = sorted(p for p in source.iterdir() if p.is_file())
    if wanted:
        images = [p for p in images if patient_id(p.name) in wanted]
    if args.limit:
        images = images[:args.limit]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for index, path in enumerate(images, start=1):
        image = Image.open(path).convert("RGB")
        candidates = propose_for_image(image, models, args.conf, args.imgsz)
        records.append({
            "image": path.name,
            "patient": patient_id(path.name),
            "width": image.width,
            "height": image.height,
            "candidates": candidates,
            "reviewed": False,
            "accepted": [],
        })
        if index % 10 == 0 or index == len(images):
            elapsed = time.time() - started
            rate = elapsed / index
            print(f"{index}/{len(images)} images; {rate:.2f}s/img; "
                  f"with_candidates={sum(bool(r['candidates']) for r in records)}", flush=True)

    payload = {
        "source": str(source.resolve()),
        "label_class": args.label_class,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "image_count": len(records),
        "images_with_candidates": sum(bool(r["candidates"]) for r in records),
        "records": records,
        "warning": "Candidate boxes are model proposals, not verified labels.",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
