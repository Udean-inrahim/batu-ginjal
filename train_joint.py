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
    data=str(DATA), imgsz=512, epochs=10, batch=16, workers=0,
    project=str(ROOT / "runs" / "kidneyai"), name="kidney-stone-coronal-balanced-v2",
    patience=5, plots=False, device="cpu", close_mosaic=2,
)
