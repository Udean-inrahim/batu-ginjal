"""Train the available Kidney localization model from KiTS23 kidney masks."""
from pathlib import Path
from ultralytics import YOLO

DATA = Path("dataset/kits23_kidney/data.yaml")
if not DATA.exists():
    raise SystemExit("Dataset belum disiapkan. Jalankan: python tools/prepare_hf_yolo.py")

model = YOLO("yolov8n.pt")
model.train(data=str(DATA.resolve()), imgsz=512, epochs=50, batch=8, workers=0,
            project=str(Path("runs/kidneyai").resolve()), name="kidney-kits23", patience=12, plots=True)
model.val(data=str(DATA.resolve()), split="test")
