from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io
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
            kidney_threshold = float(request.form.get("kidney_confidence", "0.05"))
            stone_threshold = float(request.form.get("stone_confidence", request.form.get("confidence", "0.10")))
        except ValueError:
            return jsonify(success=False, message="Ambang confidence harus berupa angka."), 400
        if not 0.05 <= kidney_threshold <= 0.9 or not 0.03 <= stone_threshold <= 0.9:
            return jsonify(success=False, message="Ambang ginjal harus 5–90% dan batu 3–90%."), 400
        detections = []
        stone_model = get_model()
        img_w, img_h = image.size
        # Scan the full frame plus overlapping tiles. Tiles give tiny stones more
        # pixels at inference time, especially in coronal/sagittal screenshots.
        stone_candidates = []
        scan_inputs = [(image, 0, 0, img_w, img_h)]
        if min(img_w, img_h) >= 384:
            mid_x, mid_y = img_w // 2, img_h // 2
            x_ranges = [(0, min(img_w, int(img_w * 0.62))),
                        (max(0, int(img_w * 0.38)), img_w)]
            y_ranges = [(0, min(img_h, int(img_h * 0.62))),
                        (max(0, int(img_h * 0.38)), img_h)]
            for left, right in x_ranges:
                for top, bottom in y_ranges:
                    scan_inputs.append((image.crop((left, top, right, bottom)), left, top, right, bottom))

        for scan_image, offset_x, offset_y, _, _ in scan_inputs:
            scan_result = stone_model.predict(scan_image, conf=min(stone_threshold, 0.08),
                                              imgsz=512, verbose=False)[0]
            for box in scan_result.boxes:
                class_id = int(box.cls.item())
                label = scan_result.names.get(class_id, str(class_id))
                if label.lower() != "stone":
                    continue
                coords = box.xyxy[0].tolist()
                mapped = [round(coords[0] + offset_x, 2), round(coords[1] + offset_y, 2),
                          round(coords[2] + offset_x, 2), round(coords[3] + offset_y, 2)]
                stone_candidates.append({"class": "Stone", "confidence": round(float(box.conf.item()), 4), "bbox": mapped})

        # Kidney class comes from the joint checkpoint, then the dedicated mask
        # model can add improved kidney boxes. This model is never used to infer stone.
        kidney_candidates = []
        full_result = stone_model.predict(image, conf=min(kidney_threshold, 0.08), imgsz=512, verbose=False)[0]
        for box in full_result.boxes:
            label = full_result.names.get(int(box.cls.item()), "")
            if label.lower() == "kidney":
                xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                kidney_candidates.append((float(box.conf.item()), xyxy))
        kidney_model_available = KIDNEY_MODEL_PATH.is_file()
        kidney_boxes = []
        if kidney_model_available:
            # Run the kidney model on the full image and overlapping left/right
            # halves. In axial slices, each kidney then occupies more of the model
            # input, helping it return both sides rather than only the clearer one.
            kidney_model = get_kidney_model()
            kidney_inputs = [(image, 0)]
            if img_w >= 320:
                split_x = img_w // 2
                overlap = int(img_w * 0.10)
                kidney_inputs.extend([
                    (image.crop((0, 0, min(img_w, split_x + overlap), img_h)), 0),
                    (image.crop((max(0, split_x - overlap), 0, img_w, img_h)), max(0, split_x - overlap)),
                ])
            for kidney_input, offset_x in kidney_inputs:
                kidney_result = kidney_model.predict(kidney_input, conf=0.01, imgsz=640, verbose=False)[0]
                for box in kidney_result.boxes:
                    coords = box.xyxy[0].tolist()
                    xyxy = [round(coords[0] + offset_x, 2), round(coords[1], 2),
                            round(coords[2] + offset_x, 2), round(coords[3], 2)]
                    kidney_candidates.append((float(box.conf.item()), xyxy))

            # candidates from joint and dedicated kidney weights are combined below

        kidney_candidates.sort(key=lambda item: item[0], reverse=True)
        displayed_kidneys = []
        for score, xyxy in kidney_candidates:
                x1, y1, x2, y2 = xyxy
                box_w, box_h = (x2 - x1) / img_w, (y2 - y1) / img_h
                center_y = ((y1 + y2) / 2) / img_h
                box_area = box_w * box_h
                # KiTS23 includes several abdominal structures around the kidneys.
                # Reject very broad masks and pelvic/liver proposals before showing
                # them as kidneys. This is a conservative ROI filter, not anatomy proof.
                if not (0.12 <= center_y <= 0.82 and 0.07 <= box_w <= 0.38
                        and 0.07 <= box_h <= 0.48 and 0.008 <= box_area <= 0.10):
                    continue
                center_x = (x1 + x2) / 2
                if any(abs(center_x - (k["bbox"][0] + k["bbox"][2]) / 2) < img_w * 0.12
                       for k in displayed_kidneys):
                    continue
                candidate = {"class": "Kidney", "confidence": round(score, 4), "bbox": xyxy}
                displayed_kidneys.append(candidate)
                kidney_boxes.append(xyxy)
                if score >= max(kidney_threshold, 0.05) and len(displayed_kidneys) <= 2:
                    detections.append(candidate)
                if len(displayed_kidneys) == 2:
                    break

        # Small calcifications can be hard to resolve at full-frame scale.
        # Run the Stone model on kidney-centered crops and map boxes to source pixels.
        if kidney_boxes and "Stone" in stone_model.names.values():
            import numpy as np
            from PIL import Image as PILImage
            for kbox in kidney_boxes:
                x1, y1, x2, y2 = kbox
                pad_x, pad_y = (x2 - x1) * 0.28, (y2 - y1) * 0.28
                left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                right, bottom = min(img_w, int(x2 + pad_x)), min(img_h, int(y2 + pad_y))
                if right - left < 24 or bottom - top < 24:
                    continue
                crop = image.crop((left, top, right, bottom))
                crop_result = stone_model.predict(crop, conf=min(stone_threshold, 0.08), imgsz=768, verbose=False)[0]
                scale_x, scale_y = (right - left) / crop.width, (bottom - top) / crop.height
                for box in crop_result.boxes:
                    if crop_result.names.get(int(box.cls.item()), "").lower() != "stone":
                        continue
                    crop_box = box.xyxy[0].tolist()
                    mapped = [
                        round(left + crop_box[0] * scale_x, 2), round(top + crop_box[1] * scale_y, 2),
                        round(left + crop_box[2] * scale_x, 2), round(top + crop_box[3] * scale_y, 2),
                    ]
                    detections.append({"class": "Stone", "confidence": round(float(box.conf.item()), 4), "bbox": mapped})

        # Supplemental detector trained on paired coronal CT Stone annotations.
        if STONE_CORONAL_MODEL_PATH.is_file():
            coronal_result = get_stone_coronal_model().predict(
                image, conf=stone_threshold, imgsz=640, verbose=False)[0]
            for box in coronal_result.boxes:
                coords = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                detections.append({"class": "Stone", "confidence": round(float(box.conf.item()), 4), "bbox": coords})

        # Suppress duplicate Stone boxes from overlapping full-frame, tile, and coronal-model predictions.
        stone_detections = sorted(
            [d for d in stone_candidates + [x for x in detections if x["class"].lower() == "stone"]
             if d["confidence"] >= stone_threshold],
            key=lambda d: d["confidence"], reverse=True)
        kidney_detections = [d for d in detections if d["class"].lower() != "stone"]
        # Stones are focal calcifications. Reject oversized boxes that span
        # bowel, spine, or pelvis, which are common pseudo-label model failures.
        plausible_stones = []
        for stone in stone_detections:
            sx1, sy1, sx2, sy2 = stone["bbox"]
            sw, sh = (sx2 - sx1) / img_w, (sy2 - sy1) / img_h
            if sw > 0.065 or sh > 0.065 or sw * sh > 0.0025:
                continue
            if kidney_detections:
                near_kidney = False
                for kidney in kidney_detections:
                    kx1, ky1, kx2, ky2 = kidney["bbox"]
                    px, py = (kx2 - kx1) * 0.35, (ky2 - ky1) * 0.35
                    if (sx2 >= kx1-px and sx1 <= kx2+px
                            and sy2 >= ky1-py and sy1 <= ky2+py):
                        near_kidney = True
                        break
                if not near_kidney:
                    continue
            plausible_stones.append(stone)
        stone_detections = plausible_stones
        unique_stones = []
        for candidate in stone_detections:
            x1, y1, x2, y2 = candidate["bbox"]
            area_a = max(1, (x2-x1)*(y2-y1))
            duplicate = False
            for kept in unique_stones:
                a1,b1,a2,b2 = kept["bbox"]
                inter = max(0,min(x2,a2)-max(x1,a1))*max(0,min(y2,b2)-max(y1,b1))
                area_b = max(1,(a2-a1)*(b2-b1))
                intersection_over_min = inter / max(1, min(area_a, area_b))
                if inter / (area_a + area_b - inter) > 0.15 or intersection_over_min > 0.55:
                    duplicate = True
                    break
            if not duplicate:
                unique_stones.append(candidate)
        detections = kidney_detections + unique_stones

        import cv2
        import numpy as np
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
        from PIL import Image as PILImage
        output = io.BytesIO()
        PILImage.fromarray(plotted).save(output, format="JPEG", quality=90)
        import base64
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
