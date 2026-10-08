"""Fine-tune the coronal stone detector for improved sensitivity on small calcifications."""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "coronal_stone" / "data.yaml"
INITIAL = ROOT / "model" / "stone_coronal_best.pt"

if not DATA.exists():
    raise SystemExit("Dataset coronal_stone tidak ditemukan.")
if not INITIAL.exists():
    raise SystemExit("model/stone_coronal_best.pt tidak ditemukan.")

print(f"Loading initial model: {INITIAL}")
model = YOLO(str(INITIAL))

print("Starting coronal stone fine-tuning...")
train_result = model.train(
    data=str(DATA),
    imgsz=640,
    epochs=25,
    batch=8,
    workers=0,
    project=str(ROOT / "runs" / "kidneyai"),
    name="coronal-stone-refined-v2",
    patience=12,
    plots=True,
    device="cpu",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.1,
    close_mosaic=5,
)

best_saved = ROOT / "runs" / "kidneyai" / "coronal-stone-refined-v2" / "weights" / "best.pt"
if best_saved.exists():
    eval_model = YOLO(str(best_saved))
    val_metrics = eval_model.val(data=str(DATA), split="test", verbose=False)
    print("\n--- HASIL EVALUASI MODEL CORONAL STONE BARU (TEST SPLIT) ---")
    print(f"Precision : {val_metrics.box.mp:.4f}")
    print(f"Recall    : {val_metrics.box.mr:.4f}")
    print(f"mAP@50    : {val_metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {val_metrics.box.map:.4f}")

    # Baseline: Precision=0.7413, Recall=0.6014, mAP50=0.6058
    if val_metrics.box.map50 >= 0.6058 or val_metrics.box.mr > 0.6014:
        print("Model baru memiliki performa lebih baik/komparatif. Memperbarui model/stone_coronal_best.pt...")
        shutil.copyfile(best_saved, INITIAL)
        print("Model diperbarui dengan sukses!")
    else:
        print("Model baru belum melampaui baseline. Model lama tetap dipertahankan.")
