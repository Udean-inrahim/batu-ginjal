"""Fine-tune the Stone detector on human-reviewed axial annotations."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "axial_reviewed" / "data.yaml"
INITIAL = ROOT / "model" / "stone_coronal_best.pt"

if not DATA.is_file() or not INITIAL.is_file():
    raise SystemExit("Dataset axial reviewed atau checkpoint Stone tidak ditemukan.")

model = YOLO(str(INITIAL))
model.train(
    data=str(DATA),
    imgsz=640,
    epochs=30,
    batch=8,
    workers=0,
    project=str(ROOT / "runs" / "kidneyai"),
    name="axial-stone-finetune-v1",
    patience=8,
    plots=False,
    device="cpu",
    close_mosaic=3,
    lr0=0.0005,
    lrf=0.01,
    freeze=10,
    degrees=7.0,
    translate=0.05,
    scale=0.15,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.4,
    mixup=0.0,
)
