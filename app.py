
import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image


# ─── Konfigurasi halaman ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Plat Nomor",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Deteksi Plat Nomor Kendaraan")
st.markdown(
    "Upload gambar kendaraan untuk mendeteksi plat nomor menggunakan **YOLOv5**."
)


# ─── Path proyek dan model ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
YOLOV5_DIR = BASE_DIR / "yolov5"


# ─── Load model YOLOv5 ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, yolov5_dir: str):
    model_file = Path(model_path)
    source_dir = Path(yolov5_dir)

    if not model_file.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_file}. "
            "Pastikan best.pt tersedia di root repository."
        )

    if not (source_dir / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"Source code YOLOv5 tidak ditemukan di {source_dir}. "
            "Pastikan folder yolov5 berisi hubconf.py dan source code "
            "YOLOv5 yang lengkap."
        )

    # Memuat checkpoint menggunakan implementasi YOLOv5 lokal.
    # CPU digunakan agar kompatibel dengan Streamlit Cloud tanpa GPU.
    model = torch.hub.load(
        str(source_dir),
        "custom",
        path=str(model_file),
        source="local",
        device="cpu",
    )

    model.eval()
    return model


# ─── Inisialisasi model ──────────────────────────────────────────────────────
with st.spinner("⏳ Memuat model YOLOv5..."):
    try:
        model = load_model(str(MODEL_PATH), str(YOLOV5_DIR))
        st.sidebar.success("✅ Model berhasil dimuat!")

    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLOv5: {e}")
        st.info(
            "Pastikan best.pt merupakan checkpoint YOLOv5 yang valid, "
            "folder yolov5 lengkap, dan dependency sesuai dengan versi "
            "model yang digunakan saat training."
        )
        st.stop()


# ─── Sidebar: Pengaturan deteksi ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05,
    )

    iou_threshold = st.slider(
        "IOU Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
    )

    img_size = st.selectbox(
        "Ukuran Input Model (px)",
        [416, 640],
        index=0,
    )

    st.markdown("---")
    st.info("**Kelas yang dideteksi:**\n\n`plat-nomor`")
    st.caption("Inference menggunakan CPU.")


# ─── Upload gambar ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📷 Upload gambar kendaraan (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)

    except Exception as e:
        st.error(f"❌ Gagal membaca gambar: {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Gambar Original")
        st.image(image_pil, use_container_width=True)

    # ─── Inferensi YOLOv5 ────────────────────────────────────────────────────
    with st.spinner("🔍 Mendeteksi plat nomor..."):
        try:
            # YOLOv5 menggunakan conf, iou, dan size.
            # Parameter imgsz merupakan format API Ultralytics yang berbeda.
            model.conf = conf_threshold
            model.iou = iou_threshold

            results = model(
                image_np,
                size=img_size,
            )

        except Exception as e:
            st.error(f"❌ Gagal melakukan inferensi: {e}")
            st.stop()

    # ─── Gambar hasil deteksi ─────────────────────────────────────────────────
    try:
        result_img = results.render()[0]

        with col2:
            st.subheader("🔍 Hasil Deteksi")
            st.image(
                result_img,
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Gagal menampilkan gambar hasil: {e}")
        st.stop()

    # ─── Tabel hasil deteksi ──────────────────────────────────────────────────
    detections = results.pandas().xyxy[0]

    if detections.empty:
        st.warning(
            "⚠️ Tidak ada plat nomor yang terdeteksi. "
            "Coba turunkan Confidence Threshold atau gunakan "
            "gambar kendaraan yang lebih jelas."
        )

    else:
        jumlah = len(detections)

        st.success(f"✅ Ditemukan **{jumlah}** objek!")

        st.subheader("📊 Detail Deteksi")

        rows = []

        for _, detection in detections.iterrows():
            rows.append(
                {
                    "Kelas": detection["name"],
                    "Confidence": f'{detection["confidence"]:.2%}',
                    "X-Min": f'{detection["xmin"]:.1f}',
                    "Y-Min": f'{detection["ymin"]:.1f}',
                    "X-Max": f'{detection["xmax"]:.1f}',
                    "Y-Max": f'{detection["ymax"]:.1f}',
                }
            )

        df_results = pd.DataFrame(rows)

        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
        )

        # ─── Ringkasan deteksi ────────────────────────────────────────────────
        st.subheader("📈 Ringkasan")

        avg_confidence = detections["confidence"].mean()

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric("Jumlah Objek", jumlah)

        with metric2:
            st.metric("Rata-rata Confidence", f"{avg_confidence:.2%}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")

st.caption(
    "Model: YOLOv5s | Dataset: Deteksi Plat Nomor (Roboflow) | "
    "mAP@0.5 ≈ 98.5%*"
)

st.caption(
    "*Nilai mAP merupakan informasi yang perlu diverifikasi "
    "berdasarkan hasil evaluasi model."
)
