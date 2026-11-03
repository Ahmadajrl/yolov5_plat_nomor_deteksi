import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
import os

# ─── Tambahkan yolov5 ke sys.path agar module-nya bisa diimpor ──────────────
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

# ─── Konfigurasi halaman ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Plat Nomor",
    page_icon="🚗",
    layout="centered",
)

# ─── Judul & deskripsi ───────────────────────────────────────────────────────
st.title("🚗 Deteksi Plat Nomor Kendaraan")
st.markdown(
    "Upload gambar kendaraan untuk mendeteksi plat nomor menggunakan **YOLOv5**."
)

# ─── Konstanta ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# ─── Load model (cache agar tidak reload setiap interaksi) ───────────────────
@st.cache_resource
def load_model(conf: float = 0.35, iou: float = 0.45):
    """Muat model YOLOv5 custom dari best.pt secara lokal."""
    model = torch.hub.load(
        YOLOV5_PATH,
        "custom",
        path=MODEL_PATH,
        source="local",
        verbose=False,
    )
    model.conf = conf
    model.iou = iou
    return model

# ─── Sidebar: pengaturan inferensi ───────────────────────────────────────────
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
    st.info(
        "**Kelas yang dideteksi:**\n\n`plat-nomor`",
    )

# ─── Muat model ──────────────────────────────────────────────────────────────
with st.spinner("⏳ Memuat model YOLOv5..."):
    try:
        model = load_model()
        # Terapkan threshold terbaru dari sidebar
        model.conf = conf_threshold
        model.iou = iou_threshold
        st.sidebar.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

# ─── Upload gambar ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📷 Upload gambar kendaraan (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Baca gambar sebagai PIL Image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Gambar Original")
        st.image(image_pil, use_column_width=True)

    # ─── Inferensi ───────────────────────────────────────────────────────────
    with st.spinner("🔍 Mendeteksi plat nomor..."):
        results = model(image_np, size=img_size)

    # ─── Render bounding box ─────────────────────────────────────────────────
    # results.render() mengembalikan list numpy array BGR
    result_img_bgr = np.array(results.render()[0])

    with col2:
        st.subheader("🔍 Hasil Deteksi")
        st.image(result_img_bgr, use_column_width=True, channels="BGR")

    # ─── Tabel hasil ─────────────────────────────────────────────────────────
    df = results.pandas().xyxy[0]

    if df.empty:
        st.warning("⚠️ Tidak ada plat nomor yang terdeteksi. Coba turunkan Confidence Threshold.")
    else:
        jumlah = len(df)
        st.success(f"✅ Ditemukan **{jumlah}** plat nomor!")

        st.subheader("📊 Detail Deteksi")
        display_df = df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].copy()
        display_df.columns = ["Kelas", "Confidence", "X-Min", "Y-Min", "X-Max", "Y-Max"]
        display_df["Confidence"] = display_df["Confidence"].map("{:.2%}".format)
        for col in ["X-Min", "Y-Min", "X-Max", "Y-Max"]:
            display_df[col] = display_df[col].map("{:.1f}".format)
        st.dataframe(display_df, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: YOLOv5s | Dataset: Deteksi Plat Nomor (Roboflow) | "
    "mAP@0.5 ≈ 98.5% | Confidence Threshold: 0.35"
)
