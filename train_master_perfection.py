"""Master refinement training to maximize Kidney and Stone localization accuracy.

Key enhancements:
1. Mosaic disabled (mosaic=0.0): Prevents cutting small stone calcifications across tile borders.
2. Loss weighting: box=8.5, cls=1.8 (boosts focal stone box tightness and class discrimination).
3. Pure Grayscale CT preservation: hsv_h=0.0, hsv_s=0.0, hsv_v=0.18 (simulates clinical CT windowing).
4. Anatomical orientation: flipud=0.0, fliplr=0.5 (strictly preserves body axes).
5. Ultra-low learning rate (lr0=0.0003, cos_lr=True) for non-destructive weight convergence.
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def refine_stone_model():
    print("=" * 65)
    print("TAHAP 1: FINE-TUNING MODEL BATU GINJAL (STONE DETECTOR)")
    print("=" * 65)
    data_yaml = ROOT / "dataset" / "coronal_stone" / "data.yaml"
    initial_weights = ROOT / "model" / "stone_coronal_best.pt"
    
    model = YOLO(str(initial_weights))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=20,
        batch=8,
        workers=0,
        project=str(ROOT / "runs" / "kidneyai"),
        name="stone-master-v4",
        patience=10,
        plots=True,
        device="cpu",
        lr0=0.0004,
        lrf=0.02,
        cos_lr=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.18,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        box=9.0,
        cls=1.8,
    )
    
    best_saved = ROOT / "runs" / "kidneyai" / "stone-master-v4" / "weights" / "best.pt"
    if best_saved.exists():
        eval_model = YOLO(str(best_saved))
        val_res = eval_model.val(data=str(data_yaml), split="test", verbose=False)
        print(f"\n[Tahap 1 Hasil Test] P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}, mAP50={val_res.box.map50:.4f}, mAP50-95={val_res.box.map:.4f}")
        if val_res.box.map50 >= 0.65 or val_res.box.mr >= 0.67:
            print("Memperbarui model/stone_coronal_best.pt dengan bobot lebih handal...")
            shutil.copyfile(best_saved, initial_weights)
            print("Model Stone berhasil diperbarui.")
        else:
            print("Bobot lama tetap dipertahankan.")

def refine_joint_model():
    print("\n" + "=" * 65)
    print("TAHAP 2: FINE-TUNING MODEL GABUNGAN (KIDNEY & STONE)")
    print("=" * 65)
    data_yaml = ROOT / "dataset" / "kidney_stone_joint" / "data.yaml"
    initial_weights = ROOT / "model" / "best.pt"
    
    model = YOLO(str(initial_weights))
    model.train(
        data=str(data_yaml),
        imgsz=512,
        epochs=6,
        batch=16,
        workers=0,
        project=str(ROOT / "runs" / "kidneyai"),
        name="joint-master-v5",
        patience=6,
        plots=True,
        device="cpu",
        lr0=0.0003,
        lrf=0.02,
        cos_lr=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        box=8.5,
        cls=1.5,
    )
    
    best_saved = ROOT / "runs" / "kidneyai" / "joint-master-v5" / "weights" / "best.pt"
    if best_saved.exists():
        eval_model = YOLO(str(best_saved))
        val_res = eval_model.val(data=str(data_yaml), split="test", verbose=False)
        print(f"\n[Tahap 2 Hasil Test] Overall P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}, mAP50={val_res.box.map50:.4f}")
        for i, c in enumerate(val_res.names.values()):
            print(f"  Kelas {c}: P={val_res.box.p[i]:.3f}, R={val_res.box.r[i]:.3f}, mAP50={val_res.box.ap50[i]:.3f}")
        if val_res.box.map50 >= 0.73 or val_res.box.mr >= 0.72:
            print("Memperbarui model/best.pt dengan bobot gabungan master...")
            shutil.copyfile(best_saved, initial_weights)
            print("Model Joint berhasil diperbarui.")
        else:
            print("Bobot lama tetap dipertahankan.")

if __name__ == "__main__":
    refine_stone_model()
    refine_joint_model()
    print("\nPelatihan master selesai dengan sukses!")
