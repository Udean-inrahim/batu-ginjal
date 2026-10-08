"""Advanced medical-CT refinement training for both Kidney and Stone detectors.

Key optimizations:
- Grayscale CT consistency: hsv_h=0, hsv_s=0, hsv_v=0.25 (simulates CT Window Width/Level variation without unrealistic color distortion)
- Anatomical orientation: flipud=0 (preserves superior-inferior anatomical axis)
- Focused loss weights: box=8.5, cls=1.5 (boosts small calcification localization)
- Cosine LR schedule with low initial learning rate for non-destructive fine-tuning
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def train_stone_model():
    print("=" * 60)
    print("FASE 1: FINE-TUNING MODEL DETEKSI BATU GINJAL (STONE)")
    print("=" * 60)
    data_yaml = ROOT / "dataset" / "coronal_stone" / "data.yaml"
    initial_weights = ROOT / "model" / "stone_coronal_best.pt"
    
    model = YOLO(str(initial_weights))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=25,
        batch=8,
        workers=0,
        project=str(ROOT / "runs" / "kidneyai"),
        name="stone-refined-v3",
        patience=12,
        plots=True,
        device="cpu",
        lr0=0.0006,
        lrf=0.02,
        cos_lr=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.1,
        close_mosaic=5,
        box=8.5,
        cls=1.5,
    )
    
    best_saved = ROOT / "runs" / "kidneyai" / "stone-refined-v3" / "weights" / "best.pt"
    if best_saved.exists():
        eval_model = YOLO(str(best_saved))
        val_res = eval_model.val(data=str(data_yaml), split="test", verbose=False)
        print(f"\n[Fase 1 Hasil] P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}, mAP50={val_res.box.map50:.4f}, mAP50-95={val_res.box.map:.4f}")
        # Baseline: mAP50=0.6665, Recall=0.6783
        if val_res.box.map50 >= 0.65 or val_res.box.mr >= 0.67:
            print("Memperbarui model/stone_coronal_best.pt dengan bobot optimal baru...")
            shutil.copyfile(best_saved, initial_weights)
            print("Model Stone berhasil diperbarui.")
        else:
            print("Bobot lama tetap dipertahankan.")

def train_joint_model():
    print("\n" + "=" * 60)
    print("FASE 2: FINE-TUNING MODEL GABUNGAN (KIDNEY & STONE)")
    print("=" * 60)
    data_yaml = ROOT / "dataset" / "kidney_stone_joint" / "data.yaml"
    initial_weights = ROOT / "model" / "best.pt"
    
    model = YOLO(str(initial_weights))
    model.train(
        data=str(data_yaml),
        imgsz=512,
        epochs=5,
        batch=16,
        workers=0,
        project=str(ROOT / "runs" / "kidneyai"),
        name="joint-refined-v4",
        patience=5,
        plots=True,
        device="cpu",
        lr0=0.0005,
        lrf=0.02,
        cos_lr=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.20,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.1,
        close_mosaic=2,
        box=8.0,
        cls=1.2,
    )
    
    best_saved = ROOT / "runs" / "kidneyai" / "joint-refined-v4" / "weights" / "best.pt"
    if best_saved.exists():
        eval_model = YOLO(str(best_saved))
        val_res = eval_model.val(data=str(data_yaml), split="test", verbose=False)
        print(f"\n[Fase 2 Hasil] Overall P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}, mAP50={val_res.box.map50:.4f}")
        for i, c in enumerate(val_res.names.values()):
            print(f"  Kelas {c}: P={val_res.box.p[i]:.3f}, R={val_res.box.r[i]:.3f}, mAP50={val_res.box.ap50[i]:.3f}")
        if val_res.box.map50 >= 0.67 or val_res.box.mr >= 0.70:
            print("Memperbarui model/best.pt dengan bobot gabungan baru...")
            shutil.copyfile(best_saved, initial_weights)
            print("Model Joint berhasil diperbarui.")
        else:
            print("Bobot lama tetap dipertahankan.")

if __name__ == "__main__":
    train_stone_model()
    train_joint_model()
    print("\nSeluruh tahapan training lanjutan selesai!")
