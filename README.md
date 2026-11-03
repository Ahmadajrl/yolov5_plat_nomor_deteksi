# 🚗 Deteksi Plat Nomor Kendaraan — YOLOv5

Aplikasi web berbasis **Streamlit** untuk mendeteksi plat nomor kendaraan secara real-time menggunakan model **YOLOv5** yang telah di-training pada dataset kustom.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![YOLOv5](https://img.shields.io/badge/YOLOv5-custom-green)
![mAP](https://img.shields.io/badge/mAP%400.5-98.5%25-brightgreen)

---

## ✨ Fitur

- 📷 Upload gambar (JPG / JPEG / PNG)
- 🔍 Deteksi plat nomor otomatis menggunakan YOLOv5
- 📊 Tampilan bounding box + tabel hasil deteksi (koordinat & confidence)
- ⚙️ Pengaturan Confidence Threshold, IOU Threshold, dan ukuran input model dari sidebar
- ⚡ Model di-cache agar inferensi cepat tanpa reload

---

## 🛠️ Instalasi

### 1. Clone repository

```bash
git clone https://github.com/<username>/deteksi_plat_nomor_yolov5.git
cd deteksi_plat_nomor_yolov5
```

### 2. Buat virtual environment (opsional tapi dianjurkan)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

**CPU Only (tanpa GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (NVIDIA CUDA):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Unduh model weights

File `best.pt` (model YOLOv5 yang sudah di-training) bisa diunduh dari:
- **GitHub Releases** halaman ini (tab *Releases*)
- Atau letakkan file `best.pt` hasil training kamu di root folder proyek

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka browser dan akses: **http://localhost:8501**

---

## 📁 Struktur Proyek

```
deteksi_plat_nomor_yolov5/
├── app.py              # Aplikasi Streamlit utama
├── best.pt             # Model YOLOv5 (Git LFS)
├── requirements.txt    # Dependency Python
├── .gitattributes      # Git LFS config untuk *.pt
├── .gitignore
├── README.md
└── yolov5/             # Source code YOLOv5 (submodule/folder)
```

---

## 🧑‍💻 Cara Penggunaan

1. Jalankan aplikasi dengan `streamlit run app.py`
2. Atur **Confidence Threshold** dan **IOU Threshold** di sidebar
3. Pilih **Ukuran Input Model** (416 atau 640 px)
4. Upload gambar kendaraan menggunakan tombol *Browse files*
5. Hasil deteksi akan tampil secara otomatis beserta tabel koordinat bounding box

---

## 📈 Performa Model

| Metric | Nilai |
|--------|-------|
| Dataset | Deteksi Plat Nomor (Roboflow) |
| Arsitektur | YOLOv5s |
| mAP@0.5 | **98.5%** |
| Confidence default | 0.35 |
| IOU default | 0.45 |

---

## 📦 Requirements

```
streamlit>=1.28.0
torch>=1.8.0
torchvision>=0.9.0
opencv-python-headless>=4.6.0
Pillow>=10.0.0
numpy>=1.23.5
pandas>=1.1.4
matplotlib>=3.3.0
PyYAML>=5.3.1
seaborn>=0.11.0
scipy>=1.4.1
```

---

## 📝 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Streamlit](https://streamlit.io/)
- [Roboflow](https://roboflow.com/) — dataset plat nomor
