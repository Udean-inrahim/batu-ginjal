from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import base64
import cv2
import io
import numpy as np
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = Path(os.getenv("KIDNEYAI_MODEL", "model/best.pt"))
STONE_CORONAL_MODEL_PATH = Path(os.getenv("KIDNEYAI_STONE_CORONAL_MODEL", "model/stone_coronal_best.pt"))
KIDNEY_MODEL_PATH = Path(os.getenv("KIDNEYAI_KIDNEY_MODEL", "model/kidney_best.pt"))
_model = None
_stone_coronal_model = None
_kidney_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


def get_stone_coronal_model():
    global _stone_coronal_model
    if _stone_coronal_model is None:
        from ultralytics import YOLO
        _stone_coronal_model = YOLO(str(STONE_CORONAL_MODEL_PATH))
    return _stone_coronal_model


def get_kidney_model():
    global _kidney_model
    if _kidney_model is None:
        from ultralytics import YOLO
        _kidney_model = YOLO(str(KIDNEY_MODEL_PATH))
    return _kidney_model


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/detect")
def detect():
    if "file" not in request.files:
        return jsonify(success=False, message="Pilih gambar CT Scan terlebih dahulu."), 400
    file = request.files["file"]
    filename = secure_filename(file.filename or "")
    if not filename or "." not in filename or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, message="Format file tidak didukung. Gunakan JPG, JPEG, atau PNG."), 400

    try:
        raw = file.read()
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(success=False, message="Gambar tidak dapat dibaca. Silakan pilih gambar lain."), 400

    if not MODEL_PATH.is_file():
        return jsonify(success=False, message="Model lokalisasi ginjal belum tersedia di model/best.pt."), 503

    try:
        try:
            kidney_threshold = float(request.form.get("kidney_confidence", "0.25"))
            stone_threshold = float(request.form.get("stone_confidence", request.form.get("confidence", "0.20")))
        except ValueError:
            return jsonify(success=False, message="Ambang confidence harus berupa angka."), 400
        if not 0.15 <= kidney_threshold <= 0.95 or not 0.05 <= stone_threshold <= 0.95:
            return jsonify(success=False, message="Ambang ginjal harus 15–95% dan batu 5–95%."), 400
        detections = []
        stone_model = get_model()
        img_w, img_h = image.size

        # Models used for stone detection across full frame and overlapping tiles.
        # Running both joint and dedicated coronal stone models on tiles provides
        # high sensitivity for small calcifications in axial, coronal, and sagittal scans.
        stone_models = [stone_model]
        if STONE_CORONAL_MODEL_PATH.is_file():
            stone_models.append(get_stone_coronal_model())

        stone_candidates = []
        scan_inputs = [(image, 0, 0)]
        if min(img_w, img_h) >= 300:
            x_ranges = [(0, min(img_w, int(img_w * 0.65))),
                        (max(0, int(img_w * 0.35)), img_w)]
            y_ranges = [(0, min(img_h, int(img_h * 0.65))),
                        (max(0, int(img_h * 0.35)), img_h)]
            for left, right in x_ranges:
                for top, bottom in y_ranges:
                    scan_inputs.append((image.crop((left, top, right, bottom)), left, top))

        for scan_image, offset_x, offset_y in scan_inputs:
            for sm in stone_models:
                scan_result = sm.predict(scan_image, conf=stone_threshold,
                                          imgsz=640, verbose=False)[0]
                for box in scan_result.boxes:
                    class_id = int(box.cls.item())
                    label = scan_result.names.get(class_id, str(class_id))
                    if label.lower() != "stone":
                        continue
                    coords = box.xyxy[0].tolist()
                    mapped = [round(coords[0] + offset_x, 2), round(coords[1] + offset_y, 2),
                              round(coords[2] + offset_x, 2), round(coords[3] + offset_y, 2)]
                    stone_candidates.append({"class": "Stone", "confidence": round(float(box.conf.item()), 4), "bbox": mapped})

        # Kidney localization from joint model and dedicated multiplane kidney weights.
        kidney_candidates = []
        full_result = stone_model.predict(image, conf=kidney_threshold, imgsz=640, verbose=False)[0]
        for box in full_result.boxes:
            label = full_result.names.get(int(box.cls.item()), "")
            if label.lower() == "kidney":
                xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                kidney_candidates.append((float(box.conf.item()), xyxy))

        kidney_model_available = KIDNEY_MODEL_PATH.is_file()
        if kidney_model_available:
            kidney_model = get_kidney_model()
            kidney_inputs = [(image, 0)]
            if img_w >= 320:
                split_x = img_w // 2
                overlap = int(img_w * 0.12)
                kidney_inputs.extend([
                    (image.crop((0, 0, min(img_w, split_x + overlap), img_h)), 0),
                    (image.crop((max(0, split_x - overlap), 0, img_w, img_h)), max(0, split_x - overlap)),
                ])
            for kidney_input, offset_x in kidney_inputs:
                kidney_result = kidney_model.predict(kidney_input, conf=kidney_threshold, imgsz=640, verbose=False)[0]
                for box in kidney_result.boxes:
                    coords = box.xyxy[0].tolist()
                    xyxy = [round(coords[0] + offset_x, 2), round(coords[1], 2),
                            round(coords[2] + offset_x, 2), round(coords[3], 2)]
                    kidney_candidates.append((float(box.conf.item()), xyxy))

        # In human anatomy, kidneys are bilateral retroperitoneal organs (1 on the right, 1 on the left).
        # The central vertical column (0.45 <= center_x <= 0.55) contains the spine, aorta, and IVC.
        # Kidneys NEVER occupy the midline spine.
        right_kidney = None  # cx < 0.48
        left_kidney = None   # cx > 0.52
        img_np = np.asarray(image)

        for score, xyxy in kidney_candidates:
            x1, y1, x2, y2 = xyxy
            box_w, box_h = (x2 - x1) / img_w, (y2 - y1) / img_h
            center_y = ((y1 + y2) / 2) / img_h
            center_x = ((x1 + x2) / 2) / img_w
            box_area = box_w * box_h

            # Reject proposals in the central spine (midline column)
            if 0.45 <= center_x <= 0.55:
                continue

            # Reject proposals too low in pelvis/sacrum or too high near chest/neck
            if center_y > 0.72 or center_y < 0.12:
                continue

            # Size sanity for kidneys across slice planes
            if not (0.05 <= box_w <= 0.45 and 0.05 <= box_h <= 0.55 and 0.005 <= box_area <= 0.20):
                continue

            # Bone density check: kidneys are soft tissue (mean intensity ~60-120).
            # Dense vertebral or pelvic bone has mean > 135 and > 30% pixels > 200.
            crop = img_np[max(0, int(y1)):min(img_h, int(y2)), max(0, int(x1)):min(img_w, int(x2))]
            if crop.size > 0:
                mean_brightness = float(crop.mean())
                dense_bone_frac = float((crop > 200).mean())
                if mean_brightness > 135 and dense_bone_frac > 0.30:
                    continue  # Reject bone/spine

            # Confidence threshold: require at least 0.25 to reject bowel/soft tissue noise
            if score < max(kidney_threshold, 0.25):
                continue

            # Assign highest confidence proposal per anatomical side (max 1 right, max 1 left)
            if center_x < 0.48:
                if right_kidney is None or score > right_kidney["confidence"]:
                    right_kidney = {"class": "Kidney", "confidence": round(score, 4), "bbox": xyxy}
            elif center_x > 0.52:
                if left_kidney is None or score > left_kidney["confidence"]:
                    left_kidney = {"class": "Kidney", "confidence": round(score, 4), "bbox": xyxy}

        displayed_kidneys = [k for k in [right_kidney, left_kidney] if k is not None]
        detections.extend(displayed_kidneys)

        # Stone localization: kidney stones MUST reside within renal parenchyma or collecting system.
        # They never reside in posterior ribs, back muscles, or spinal canal.
        valid_renal_zones = []
        midline = img_w / 2
        for kidney in displayed_kidneys:
            kx1, ky1, kx2, ky2 = kidney["bbox"]
            # Tight 6% margin around detected kidney for renal hilum
            mx, my = (kx2 - kx1) * 0.06, (ky2 - ky1) * 0.06
            valid_renal_zones.append((kx1 - mx, ky1 - my, kx2 + mx, ky2 + my))

        # If only one kidney was detected, synthesize the contralateral kidney fossa symmetrically
        if len(displayed_kidneys) == 1:
            kx1, ky1, kx2, ky2 = displayed_kidneys[0]["bbox"]
            mx, my = (kx2 - kx1) * 0.06, (ky2 - ky1) * 0.06
            ckx1 = max(0, 2 * midline - kx2 - mx)
            ckx2 = min(img_w, 2 * midline - kx1 + mx)
            valid_renal_zones.append((ckx1, ky1 - my, ckx2, ky2 + my))

        # Suppress duplicate Stone boxes and filter false positives (bones/ribs)
        stone_detections = sorted(
            [d for d in stone_candidates if d["confidence"] >= stone_threshold],
            key=lambda d: d["confidence"], reverse=True)

        plausible_stones = []
        for stone in stone_detections:
            sx1, sy1, sx2, sy2 = stone["bbox"]
            sw, sh = (sx2 - sx1) / img_w, (sy2 - sy1) / img_h
            scx = (sx1 + sx2) / 2
            scy = (sy1 + sy2) / 2

            # Reject oversized proposals (stones are focal calcifications)
            if sw > 0.08 or sh > 0.08 or sw * sh > 0.003:
                continue

            # If renal zones are localized, stone center MUST be inside a valid renal zone!
            # This definitively eliminates posterior ribs and back musculature.
            if valid_renal_zones:
                in_renal_zone = False
                for zx1, zy1, zx2, zy2 in valid_renal_zones:
                    if zx1 <= scx <= zx2 and zy1 <= scy <= zy2:
                        in_renal_zone = True
                        break
                if not in_renal_zone:
                    continue  # Rib, spine, or chest wall -> REJECT!
            else:
                # If no kidney localized, reject midline spine and body perimeter
                scx_norm = scx / img_w
                scy_norm = scy / img_h
                if 0.44 <= scx_norm <= 0.56 or not (0.15 <= scy_norm <= 0.82):
                    continue

            plausible_stones.append(stone)

        stone_detections = plausible_stones
        unique_stones = []
        for candidate in stone_detections:
            x1, y1, x2, y2 = candidate["bbox"]
            area_a = max(1, (x2 - x1) * (y2 - y1))
            cx_a, cy_a = (x1 + x2) / 2, (y1 + y2) / 2
            duplicate = False
            for kept in unique_stones:
                a1, b1, a2, b2 = kept["bbox"]
                inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
                area_b = max(1, (a2 - a1) * (b2 - b1))
                cx_b, cy_b = (a1 + a2) / 2, (b1 + b2) / 2
                dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                if (inter / (area_a + area_b - inter) > 0.15 
                        or inter / min(area_a, area_b) > 0.35 
                        or dist < max(img_w, img_h) * 0.025):
                    duplicate = True
                    break
            if not duplicate:
                unique_stones.append(candidate)
        detections = displayed_kidneys + unique_stones

        plotted = np.asarray(image).copy()
        colors = {"Kidney": (220, 70, 35), "Stone": (40, 200, 220)}
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
            color = colors[detection["class"]]
            cv2.rectangle(plotted, (x1, y1), (x2, y2), color, 2)
            label = f"{detection['class']} {detection['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(0, y1 - text_height - baseline - 4)
            cv2.rectangle(plotted, (x1, top), (x1 + text_width + 6, y1), color, -1)
            cv2.putText(plotted, label, (x1 + 3, max(text_height, y1 - baseline - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        output = io.BytesIO()
        Image.fromarray(plotted).save(output, format="JPEG", quality=90)
        return jsonify(success=True, message="Analisis selesai.", detections=detections,
                       total=len(detections), width=image.width, height=image.height,
                       confidence_threshold=stone_threshold,
                       thresholds={"kidney": kidney_threshold, "stone": stone_threshold},
                       models={"kidney": kidney_model_available,
                               "stone": "Stone" in stone_model.names.values() or STONE_CORONAL_MODEL_PATH.is_file()},
                       result_image="data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode())
    except Exception:
        app.logger.exception("Inference gagal")
        return jsonify(success=False, message="Terjadi kesalahan saat analisis. Periksa konfigurasi model lalu coba lagi."), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(success=False, message="Ukuran file maksimal 16MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
