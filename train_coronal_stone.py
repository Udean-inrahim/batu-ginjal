"""Train a focused Stone detector on paired coronal CT annotations."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "coronal_stone" / "data.yaml"
INITIAL = ROOT / "model" / "best.pt"
if not DATA.exists():
    raise SystemExit("Jalankan tools/prepare_coronal_stone.py lebih dulu.")
if not INITIAL.exists():
    raise SystemExit("model/best.pt tidak ditemukan.")

# Start from KidneyAI weights; Ultralytics remaps the matching Stone class to
# this focused one-class dataset and retains transferable visual features.
model = YOLO(str(INITIAL))
model.train(data=str(DATA), imgsz=640, epochs=80, batch=8, workers=0,
            project=str(ROOT / "runs" / "kidneyai"), name="coronal-stone-supervised",
            patience=18, plots=False, device="cpu", close_mosaic=8, cls_remap=True)
