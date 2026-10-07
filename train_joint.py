"""Refine the two-class research detector from the current best checkpoint."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "kidney_stone_joint" / "data.yaml"
INITIAL = ROOT / "model" / "best.pt"
if not DATA.exists():
    raise SystemExit("Dataset joint tidak ada. Jalankan tools/combine_kidney_stone.py dulu.")
if not INITIAL.exists():
    raise SystemExit("Checkpoint model/best.pt tidak ditemukan.")

"""Refine the two-class research detector from the current best checkpoint."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "kidney_stone_joint" / "data.yaml"
INITIAL = ROOT / "model" / "best.pt"
if not DATA.exists():
    raise SystemExit("Dataset joint tidak ada. Jalankan tools/combine_kidney_stone.py dulu.")
if not INITIAL.exists():
    raise SystemExit("Checkpoint model/best.pt tidak ditemukan.")

model = YOLO(str(INITIAL))
model.train(
    data=str(DATA),
    imgsz=640,
    epochs=50,
    batch=8,
    workers=0,
    project=str(ROOT / "runs" / "kidneyai"),
    name="joint-v3-robust",
    patience=12,
    plots=False,
    device="cpu",
    close_mosaic=5,
    lr0=0.001,
    lrf=0.01,
    degrees=10.0,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    flipud=0.1,
    mosaic=1.0,
    mixup=0.1,
)
