# KidneyAI

Prototype web berbahasa Indonesia untuk meninjau citra CT. Aplikasi Flask menjalankan model YOLO dan menampilkan kotak, label, confidence, serta gambar anotasi.

> **Bukan alat diagnosis.** Hasil harus diverifikasi oleh tenaga medis profesional. Model Stone menggunakan campuran anotasi dataset publik dan pseudo-label Mask R-CNN; skor bukan probabilitas diagnosis dan bukan validasi klinis.

## Menjalankan di Windows

Gunakan Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000. Unggah PNG/JPG/JPEG hingga 16 MB. Server harus tetap berjalan di terminal selama situs digunakan.

## Model inference

Bobot yang disertakan:

- `model/best.pt`: detektor gabungan dengan kelas `Kidney` dan `Stone`.
- `model/kidney_best.pt`: model lokalisasi ginjal multi-plane.
- `model/stone_coronal_best.pt`: model tambahan Stone yang dilatih khusus pada CT koronal.

Lokasi checkpoint dapat diubah dengan environment variable `KIDNEYAI_MODEL`, `KIDNEYAI_KIDNEY_MODEL`, dan `KIDNEYAI_STONE_CORONAL_MODEL`.

## Evaluasi dan batasan

Model Kidney dilatih menggunakan kotak yang dikonversi dari mask segmentasi KiTS23 dan split berdasarkan kasus. Model Stone koronal memakai 62 pasangan CT/label untuk train/validasi dan menyisihkan 77 gambar sebagai test. Hasil test koronal model khusus: precision 0.741, recall 0.601, mAP@50 0.606, mAP@50:95 0.259. Karena jumlah data sedikit dan sebagian label Stone berasal dari prediksi model lain, model dapat melewatkan batu atau memberi false positive, terutama pada ukuran, orientasi, kontras, dan protokol scan yang berbeda.

Output `0 objek` tidak memastikan bahwa batu tidak ada. Output `Stone` adalah kandidat untuk ditinjau radiolog, bukan hasil diagnosis.

## API

`POST /api/detect` menerima multipart field `file`, `kidney_confidence`, dan `stone_confidence` (ambang opsional). Respons memuat `detections` berisi `class`, `confidence`, dan `bbox`, ukuran gambar, status model, ambang, serta `result_image` JPEG data URL. File tidak disimpan sebagai riwayat.

## Dataset dan training

Dataset mentah, cache, dan run training diabaikan Git melalui `.gitignore`; script di `tools/` menyiapkan ulang data. Training scripts: `train.py`, `train_kidney_multiplane.py`, `train_coronal_stone.py`, dan `train_joint.py`.
