"""Build a plane-balanced kidney box dataset from the KiTS23 2D slice API."""
from pathlib import Path
import collections
import io

import cv2
import numpy as np
import requests
from PIL import Image

API = "https://datasets-server.huggingface.co/rows"
DATASET = "ryfkn/KiTS23-sliced-2d"
OUT = Path(__file__).resolve().parents[1] / "dataset" / "kits23_kidney_coronal"
MIN_AREA = 120
PLAN = {"train": ["case_00101", "case_00156"], "val": ["case_00442"], "test": ["case_00156"]}
QUOTA = {"train": 240, "val": 90, "test": 60}
NEEDED = set(PLAN["train"]) | set(PLAN["val"]) | set(PLAN["test"])

session = requests.Session()


def index_rows():
    rows = []
    seen = set()
    offset = 0
    while offset < 6000 and not NEEDED <= seen:
        response = session.get(API, params={"dataset": DATASET, "config": "default",
                                            "split": "train", "offset": offset, "length": 100}, timeout=90)
        if response.status_code != 200:
            print(f"index berhenti di offset {offset}: HTTP {response.status_code}")
            break
        for item in response.json()["rows"]:
            rows.append((item["row_idx"], item["row"]))
            seen.add(item["row"]["case_id"])
        offset += 100
    return rows


def boxes_from_mask(mask_image):
    array = np.asarray(mask_image) > 127
    if not array.any():
        return []
    height, width = array.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(array.astype("uint8"), 8)
    boxes = []
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component].tolist()
        if area >= MIN_AREA:
            boxes.append((x, y, x + box_width, y + box_height))
    if not boxes:
        ys, xs = np.where(array)
        boxes = [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]
    normalized = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append((((x1 + x2) / 2) / width,
                           ((y1 + y2) / 2) / height,
                           (x2 - x1) / width,
                           (y2 - y1) / height))
    return normalized


def main():
    rows = index_rows()
    lookup = dict(rows)
    by_case_axis = collections.defaultdict(list)
    for index, row in rows:
        by_case_axis[(row["case_id"], row["axis"])].append(index)

    for split in PLAN:
        for kind in ("images", "labels"):
            target = OUT / kind / split
            target.mkdir(parents=True, exist_ok=True)
            for old in target.glob("*"):
                if old.is_file():
                    old.unlink()

    written = collections.Counter()
    for split, cases in PLAN.items():
        per_plane = QUOTA[split] // 3
        for case in cases:
            for axis in (2, 0, 1):
                candidates = sorted(by_case_axis.get((case, axis), []), reverse=True)
                for index in candidates[:per_plane]:
                    row = lookup[index]
                    try:
                        image = Image.open(io.BytesIO(
                            session.get(row["slice_image_ct"]["src"], timeout=60).content)).convert("RGB")
                        mask = Image.open(io.BytesIO(
                            session.get(row["slice_kidney_mask"]["src"], timeout=60).content)).convert("L")
                    except Exception:
                        continue
                    boxes = boxes_from_mask(mask)
                    if not boxes:
                        continue
                    stem = f"{row['case_id']}_a{axis}_s{row['slice_idx']:04d}"
                    image.save(OUT / "images" / split / f"{stem}.jpg", quality=95)
                    lines = ["0 %.6f %.6f %.6f %.6f" % box for box in boxes]
                    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                    written[f"{split}_a{axis}"] += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: Kidney\n", encoding="utf-8")
    print(dict(sorted(written.items())))
    print("yaml:", OUT / "data.yaml")


if __name__ == "__main__":
    main()
