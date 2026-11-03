import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

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

# ─── Path model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# ─── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


with st.spinner("⏳ Memuat model YOLOv5..."):
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    conf_threshold = st.slider(
        "Confidence Threshold", min_value=0.10, max_value=1.0, value=0.35, step=0.05
    )
    iou_threshold = st.slider(
        "IOU Threshold", min_value=0.10, max_value=1.0, value=0.45, step=0.05
    )
    img_size = st.selectbox("Ukuran Input Model (px)", [416, 640], index=0)
    st.markdown("---")
    st.info("**Kelas yang dideteksi:**\n\n`plat-nomor`")

# ─── Upload gambar ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📷 Upload gambar kendaraan (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Gambar Original")
        st.image(image_pil, use_column_width=True)

    with st.spinner("🔍 Mendeteksi plat nomor..."):
        try:
            results = model(
                image_np,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=img_size,
                verbose=False,
            )
        except Exception as e:
            st.error(f"❌ Gagal melakukan inferensi: {e}")
            st.stop()

    # ─── Gambar hasil ─────────────────────────────────────────────────────────
    result_img_bgr = results[0].plot()
    with col2:
        st.subheader("🔍 Hasil Deteksi")
        st.image(result_img_bgr, use_column_width=True, channels="BGR")

    # ─── Tabel hasil ─────────────────────────────────────────────────────────
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        st.warning(
            "⚠️ Tidak ada plat nomor yang terdeteksi. "
            "Coba turunkan Confidence Threshold."
        )
    else:
        jumlah = len(boxes)
        st.success(f"✅ Ditemukan **{jumlah}** plat nomor!")
        st.subheader("📊 Detail Deteksi")

        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        rows = [
            {
                "Kelas":       names[cls_ids[i]],
                "Confidence":  f"{confs[i]:.2%}",
                "X-Min":       f"{xyxy[i][0]:.1f}",
                "Y-Min":       f"{xyxy[i][1]:.1f}",
                "X-Max":       f"{xyxy[i][2]:.1f}",
                "Y-Max":       f"{xyxy[i][3]:.1f}",
            }
            for i in range(jumlah)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: YOLOv5s | Dataset: Deteksi Plat Nomor (Roboflow) | "
    "mAP@0.5 ≈ 98.5% | Confidence Threshold: 0.35"
)
