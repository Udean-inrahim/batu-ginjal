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

        kidney_candidates.sort(key=lambda item: item[0], reverse=True)
        displayed_kidneys = []
        for score, xyxy in kidney_candidates:
            x1, y1, x2, y2 = xyxy
            box_w, box_h = (x2 - x1) / img_w, (y2 - y1) / img_h
            center_y = ((y1 + y2) / 2) / img_h
            center_x = ((x1 + x2) / 2) / img_w
            box_area = box_w * box_h
            if not (0.05 <= center_y <= 0.95 and 0.03 <= box_w <= 0.55
                    and 0.03 <= box_h <= 0.65 and 0.002 <= box_area <= 0.25):
                continue
            
            # Exclusion of posterior midline objects (vertebra/spine)
            # Kidneys are strictly bilateral (retroperitoneal), never sitting exactly in the midline
            # In axial/coronal CT, the spine (midline) occupies ~ 0.40 to 0.60.
            if 0.40 <= center_x <= 0.60:
                continue

            if any(abs(center_x - (k["bbox"][0] + k["bbox"][2]) / (2 * img_w)) < 0.10
                   for k in displayed_kidneys):
                continue
            if score >= max(kidney_threshold, 0.15):
                candidate = {"class": "Kidney", "confidence": round(score, 4), "bbox": xyxy}
                displayed_kidneys.append(candidate)
                detections.append(candidate)
            if len(displayed_kidneys) == 2:
                break

        # Suppress duplicate Stone boxes and filter false positives (bones/ribs)
        stone_detections = sorted(
            [d for d in stone_candidates if d["confidence"] >= stone_threshold],
            key=lambda d: d["confidence"], reverse=True)
        kidney_detections = [d for d in detections if d["class"].lower() != "stone"]

        plausible_stones = []
        for stone in stone_detections:
            sx1, sy1, sx2, sy2 = stone["bbox"]
            sw, sh = (sx2 - sx1) / img_w, (sy2 - sy1) / img_h
            scx, scy = (sx1 + sx2) / (2 * img_w), (sy1 + sy2) / (2 * img_h)
            if not (0.10 <= scx <= 0.90 and 0.15 <= scy <= 0.88):
                continue

            crop_x1, crop_y1 = int(max(0, sx1)), int(max(0, sy1))
            crop_x2, crop_y2 = int(min(img_w, sx2)), int(min(img_h, sy2))
            if crop_x2 - crop_x1 < 1 or crop_y2 - crop_y1 < 1:
                continue
            crop = np.asarray(image)[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue
            median_brightness = np.median(crop)
            std_brightness = np.std(crop)
            is_dense_focal = (median_brightness > 150) and (std_brightness < 75)

            if sw > 0.25 or sh > 0.25 or sw * sh > 0.045:
                continue
            if sw < 0.02 or sh < 0.02 or sw * sh < 0.0002:
                if not is_dense_focal:
                    continue

            if kidney_detections:
                near_kidney = False
                for kidney in kidney_detections:
                    kx1, ky1, kx2, ky2 = kidney["bbox"]
                    px, py = (kx2 - kx1) * 0.15, (ky2 - ky1) * 0.15
                    if (sx2 >= kx1 - px and sx1 <= kx2 + px
                            and sy2 >= ky1 - py and sy1 <= ky2 + py):
                        near_kidney = True
                        break
                    midline = img_w / 2
                    ckx1 = max(0, 2 * midline - kx2 - px)
                    ckx2 = min(img_w, 2 * midline - kx1 + px)
                    if (sx2 >= ckx1 and sx1 <= ckx2 and sy2 >= ky1 - py and sy1 <= ky2 + py):
                        near_kidney = True
                        break
                if not near_kidney:
                    continue
            else:
                if not is_dense_focal and median_brightness < 180:
                    continue
            plausible_stones.append(stone)

        stone_detections = plausible_stones
        unique_stones = []
        for candidate in stone_detections:
            x1, y1, x2, y2 = candidate["bbox"]
            area_a = max(1, (x2 - x1) * (y2 - y1))
            duplicate = False
            for kept in unique_stones:
                a1, b1, a2, b2 = kept["bbox"]
                inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
                area_b = max(1, (a2 - a1) * (b2 - b1))
                intersection_over_min = inter / max(1, min(area_a, area_b))
                if inter / (area_a + area_b - inter) > 0.18 or intersection_over_min > 0.50:
                    duplicate = True
                    break
            if not duplicate:
                unique_stones.append(candidate)
        detections = kidney_detections + unique_stones

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
