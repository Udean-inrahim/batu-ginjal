"""Fine-tune kidney localization across axial, coronal, and sagittal CT slices."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "kits23_kidney_coronal" / "data.yaml"
INITIAL = ROOT / "runs" / "kidneyai" / "kidney-multiplane-2" / "weights" / "best.pt"
if not DATA.exists():
    raise SystemExit("Dataset multi-plane belum disiapkan: tools/prepare_kidney_coronal.py")
if not INITIAL.exists():
    raise SystemExit("Checkpoint kidney awal tidak ditemukan.")

model = YOLO(str(INITIAL))
model.train(data=str(DATA), imgsz=512, epochs=6, batch=8, workers=0,
            project=str(ROOT / "runs" / "kidneyai"), name="kidney-multiplane-refined",
            patience=3, plots=False, device="cpu", close_mosaic=2)
