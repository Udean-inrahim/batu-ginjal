"""Download patient-separated kidney masks and convert them to YOLO boxes."""
from pathlib import Path
import json
import random
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "kits23_kidney"
API = "https://datasets-server.huggingface.co/rows"
DATASET = "ryfkn/KiTS23-sliced-2d"
TRAIN_CASES = 6
VAL_CASES = 2
TEST_CASES = 2
ROWS_PER_PAGE = 100
MAX_SLICES_PER_SPLIT = {"train": 900, "val": 180, "test": 180}
MIN_MASK_AREA = 120
session = requests.Session()


def fetch_page(split, offset, length):
    # The public rows API rate-limits aggressive indexing, so back off and retry.
    last_error = None
    for attempt in range(5):
        response = session.get(API, params={"dataset": DATASET, "config": "default",
                                           "split": split, "offset": offset, "length": length}, timeout=90)
        if response.status_code == 200:
            return response.json()
        last_error = response.status_code
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gagal mengambil baris {split}@{offset} (HTTP {last_error}).")


def main():
    first = fetch_page("train", 0, ROWS_PER_PAGE)
    total = first["num_rows_total"]
    indexed_rows = []
    cases = []
    for offset in range(0, total, ROWS_PER_PAGE):
        page = first if offset == 0 else fetch_page("train", offset, ROWS_PER_PAGE)
        for item in page["rows"]:
            indexed_rows.append(item)
            case = item["row"]["case_id"]
            if not cases or cases[-1] != case:
                cases.append(case)
        if len(cases) >= 40:
            break
    # Rows are grouped by case, then by plane (axis 0 axial, 1 sagittal, 2 coronal).
    # Group by case and plane so every split contains a mix of planes.
    case_plane_rows = {}
    for item in indexed_rows:
        row = item["row"]
        case_plane_rows.setdefault((row["case_id"], row["axis"]), []).append(item["row_idx"])
    rng = random.Random(2306)
    rng.shuffle(cases)
    # Coronal planes are the rarest in this dataset, so prefer cases that contain
    # axis 2 and fall back to any case that has kidney-bearing slices.
    coronal_cases = [case for case in cases if any(
        row_case == case and axis == 2 for (row_case, axis) in case_plane_rows)]
    cases = coronal_cases if len(coronal_cases) >= TRAIN_CASES + VAL_CASES + TEST_CASES else cases
    split_sizes = {"train": TRAIN_CASES, "val": VAL_CASES, "test": TEST_CASES}
    splits = {}
    cursor = 0
    for split, size in split_sizes.items():
        splits[split] = set(cases[cursor:cursor + size])
        cursor += size
    selected_indices = set()
    for split, case_set in splits.items():
        quota = MAX_SLICES_PER_SPLIT[split]
        plane_pools = {plane: [] for plane in (0, 1, 2)}
        for case in case_set:
            for (row_case, axis), rows in case_plane_rows.items():
                if row_case == case:
                    plane_pools[axis].extend(rows)
        # Reserve a third of the budget for coronal slices so kidney training
        # covers the plane used by most uploaded screenshots.
        shares = {2: quota // 3, 0: quota // 3, 1: quota - 2 * (quota // 3)}
        for plane in (2, 0, 1):
            pool = plane_pools[plane]
            rng.shuffle(pool)
            selected_indices.update(pool[:shares[plane]])
    print({"case_split": {key: sorted(value) for key, value in splits.items()},
           "selected_slices": len(selected_indices)}, flush=True)

    for split in splits:
        for kind in ("images", "labels"):
            target = OUT / kind / split
            target.mkdir(parents=True, exist_ok=True)
            for old in target.glob("*"):
                if old.is_file():
                    old.unlink()

    done = {"train": 0, "val": 0, "test": 0}
    processed = set()
    for page_id in sorted({idx // ROWS_PER_PAGE for idx in selected_indices}):
        page = next((p for p in [first] if page_id == 0), None)
        if page is None:
            page = fetch_page("train", page_id * ROWS_PER_PAGE, ROWS_PER_PAGE)
        for item in page["rows"]:
            idx, row = item["row_idx"], item["row"]
            if idx not in selected_indices or idx in processed:
                continue
            split = next(key for key, values in splits.items() if row["case_id"] in values)
            try:
                image = Image.open(BytesIO(session.get(row["slice_image_ct"]["src"], timeout=60).content)).convert("RGB")
                mask = Image.open(BytesIO(session.get(row["slice_kidney_mask"]["src"], timeout=60).content)).convert("L")
                mask_array = np.asarray(mask) > 127
                height, width = mask_array.shape
                labels = []
                # Separate left/right kidneys where mask connected components permit it.
                import cv2
                count, components, stats, _ = cv2.connectedComponentsWithStats(mask_array.astype("uint8"), 8)
                boxes = []
                for component_id in range(1, count):
                    x, y, box_width, box_height, area = stats[component_id].tolist()
                    if area >= MIN_MASK_AREA:
                        boxes.append((x, y, x + box_width, y + box_height))
                if not boxes and mask_array.any():
                    ys, xs = np.where(mask_array)
                    boxes = [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]
                for x1, y1, x2, y2 in boxes:
                    xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                    bw, bh = (x2 - x1) / width, (y2 - y1) / height
                    labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stem = f"{row['case_id']}_a{row['axis']}_s{row['slice_idx']:04d}"
                image.save(OUT / "images" / split / f"{stem}.jpg", quality=95)
                (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
                done[split] += 1
                processed.add(idx)
                if len(processed) % 100 == 0:
                    print(f"slices {len(processed)}/{len(selected_indices)}", flush=True)
            except Exception as exc:
                print(f"skip row {idx}: {exc}", flush=True)

    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: Kidney\n", encoding="utf-8")
    (OUT / "source.json").write_text(json.dumps({"dataset": DATASET, "counts": done,
        "case_split": {k: sorted(v) for k, v in splits.items()}, "labels": "Kidney mask converted to bbox"}, indent=2), encoding="utf-8")
    print({"counts": done, "yaml": str(data_yaml)})


if __name__ == "__main__":
    main()
