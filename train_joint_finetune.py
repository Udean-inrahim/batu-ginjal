"""Fine-tune the two-class joint model (Kidney + Stone) for balanced accuracy."""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "kidney_stone_joint" / "data.yaml"
INITIAL = ROOT / "model" / "best.pt"

if not DATA.exists():
    raise SystemExit("Dataset kidney_stone_joint tidak ditemukan.")
if not INITIAL.exists():
    raise SystemExit("model/best.pt tidak ditemukan.")

print(f"Memuat model gabungan awal: {INITIAL}")
model = YOLO(str(INITIAL))

print("Memulai fine-tuning joint detector (Kidney & Stone)...")
model.train(
    data=str(DATA),
    imgsz=512,
    epochs=5,
    batch=16,
    workers=0,
    project=str(ROOT / "runs" / "kidneyai"),
    name="joint-refined-v3",
    patience=5,
    plots=True,
    device="cpu",
    lr0=0.0008,
    lrf=0.01,
    cos_lr=True,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.1,
    close_mosaic=2,
)

best_saved = ROOT / "runs" / "kidneyai" / "joint-refined-v3" / "weights" / "best.pt"
if best_saved.exists():
    eval_model = YOLO(str(best_saved))
    val_metrics = eval_model.val(data=str(DATA), split="test", verbose=False)
    print("\n--- HASIL EVALUASI MODEL JOINT BARU (TEST SPLIT) ---")
    print(f"Overall Precision : {val_metrics.box.mp:.4f}")
    print(f"Overall Recall    : {val_metrics.box.mr:.4f}")
    print(f"Overall mAP@50    : {val_metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95 : {val_metrics.box.map:.4f}")
    for i, c in enumerate(val_metrics.names.values()):
        print(f"  Kelas {c}: P={val_metrics.box.p[i]:.3f}, R={val_metrics.box.r[i]:.3f}, mAP50={val_metrics.box.ap50[i]:.3f}")

    # Baseline: mAP50=0.6767, Stone mAP50=0.452
    stone_ap50 = val_metrics.box.ap50[1] if len(val_metrics.box.ap50) > 1 else 0
    if val_metrics.box.map50 >= 0.6767 or stone_ap50 > 0.452:
        print("Model baru menunjukkan performa lebih baik/komparatif. Memperbarui model/best.pt...")
        shutil.copyfile(best_saved, INITIAL)
        print("Model model/best.pt diperbarui dengan sukses!")
    else:
        print("Model baru belum melampaui baseline. Model lama tetap dipertahankan.")
