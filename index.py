import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import shutil
import os
import plotly.express as px
import io
import base64
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
import tempfile
import time
import tempfile, os
import plotly.io as pio
from pptx import Presentation
from pptx.util import Inches

RESET_FLAG = "reset_flag.txt"


def preprocess_period_column(df):
    """Mempersiapkan kolom 'Periode' agar konsisten sebagai datetime (awal bulan)."""
    df = df.copy()
    if "Periode" not in df.columns:
        raise ValueError("Kolom 'Periode' tidak ditemukan.")
    # Parse ke datetime
    df["Periode"] = pd.to_datetime(df["Periode"], errors="coerce", infer_datetime_format=True)
    # Jika masih ada NaT, coba format-format umum (beberapa dataset menuliskan dd/mm/YYYY atau YYYY-MM)
    if df["Periode"].isna().any():
        for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%Y-%m", "%Y-%m-%d", "%d-%m-%Y"):
            parsed = pd.to_datetime(df["Periode"].astype(str), format=fmt, errors="coerce")
            df["Periode"] = df["Periode"].fillna(parsed)
            if df["Periode"].notna().all():
                break

    df = df.dropna(subset=["Periode"]).copy()


    df["Periode"] = df["Periode"].dt.to_period("M").dt.to_timestamp()

    if "Pemasukan" in df.columns:
        df["Pemasukan"] = pd.to_numeric(df["Pemasukan"], errors="coerce")
        df = df.dropna(subset=["Pemasukan"])

    df = df.sort_values("Periode").reset_index(drop=True)
    return df



# ============================================================
# Setup halaman
# ============================================================
st.set_page_config(page_title="RevFlux", page_icon="Logo.png", layout="wide")

logo_path = Path("Logo.png")
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="width:28px;height:28px;margin-right:10px;">'
else:
    logo_html = "<div style='width:28px;height:28px;background:#ccc;border-radius:50%;margin-right:10px;'></div>"

navbar_css = """
<style>
header[data-testid="stHeader"], footer, div[data-testid="stToolbar"] {display:none;}
.navbar {
  position: fixed; top: 0; left: 0; width: 100%; height: 60px;
  display: flex; justify-content: flex-start; align-items: center;
  padding: 0 24px; z-index: 9999; border-radius: 0 0 10px 10px;
  background-color: #f8f9fa; color: #333;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.navbar span {font-weight:700;font-size:20px;color:#8e44ad;}
.block-container {padding-top:80px !important;}
.divider {border:none;height:2px;margin:30px 0 20px 0;background:linear-gradient(to right, #f8f9fa);}
div.stDownloadButton>button, div.stButton>button {
    border-radius:8px !important; font-weight:600 !important;
}
div.stButton>button:has(span:contains('Train Model')),
div.stButton>button:has(span:contains('Retrain Model')),
div.stDownloadButton>button {
    background-color:#8e44ad !important; color:white !important; border:none !important;
}
div.stDownloadButton>button:hover, div.stButton>button:hover {opacity:0.9 !important;}
div.stButton>button:has(span:contains('Reset Model')) {
    background-color:#f39c12 !important; color:white !important;
}
div.stButton>button:has(span:contains('Kembalikan Model')) {
    background-color:#3498db !important; color:white !important;
}
</style>
"""
st.markdown(navbar_css, unsafe_allow_html=True)

navbar_html = f"<div class='navbar'>{logo_html}<span>RevFlux</span></div>"
st.markdown(navbar_html, unsafe_allow_html=True)


# ============================================================
# Pembuka
# ============================================================
st.markdown( """ <div style='text-align: center;'> <h1 style='font-size:40px; color:#8e44ad; font-weight:800;'>RevFlux</h1> <p style='font-size:22px; font-weight:600; color:#333;'> Prediksi Penjualan Cerdas Berbasis <span style='color:#8e44ad;'>Machine Learning</span> </p> <hr style='margin-top:20px; margin-bottom:20px; border: 1px solid #8e44ad;'/> <p style='font-size:17px; color:#555; max-width:700px; margin:auto;'> RevFlux membantu Anda menganalisis tren penjualan dan memprediksi pendapatan masa depan menggunakan teknologi kecerdasan buatan. Unggah data Anda, latih model, dan temukan wawasan baru yang dapat mendukung keputusan bisnis Anda. 📈 </p> </div> """, unsafe_allow_html=True )


# ============================================================
# Upload File & Setup Dataset
# ============================================================
st.subheader("📂 Input Data Baru")
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload data penjualan (CSV/XLSX, kolom: Periode & Pemasukan)", type=["csv", "xlsx"])

contoh_data = pd.DataFrame({
    "Periode": pd.date_range("2025-01-01", periods=4, freq="MS").strftime("%Y-%m-%d"),
    "Pemasukan": [12000000, 13500000, 15000000, 14500000]
})
csv_buffer = io.StringIO()
contoh_data.to_csv(csv_buffer, index=False)
st.download_button("⬇️ Download Template Data (CSV)", csv_buffer.getvalue(), "contoh_data.csv", "text/csv")


active_dataset = None
if os.path.exists("active_dataset.txt"):
    with open("active_dataset.txt") as f:
        active_dataset = f.read().strip()

# ============================================================
# Proses Upload
# ============================================================
if uploaded_file:
    st.session_state["uploaded_file_name"] = uploaded_file.name   # <<== SESSION

    current_dataset_name = active_dataset
    if current_dataset_name is None:
        current_dataset_name = Path(uploaded_file.name).stem
        with open("active_dataset.txt", "w") as f:
            f.write(current_dataset_name)
        active_dataset = current_dataset_name

    st.session_state["active_dataset"] = active_dataset           # <<== SESSION

    data_filename = f"{current_dataset_name}_data.csv"

    try:
        if uploaded_file.name.endswith(".csv"):
            new_data = pd.read_csv(uploaded_file)
        else:
            new_data = pd.read_excel(uploaded_file)

        if "Periode" not in new_data.columns or "Pemasukan" not in new_data.columns:
            st.error("❌ File harus memiliki kolom 'Periode' dan 'Pemasukan'.")
            st.stop()

        new_data = preprocess_period_column(new_data)

        if os.path.exists(data_filename):
            old_data = pd.read_csv(data_filename)
            old_data = preprocess_period_column(old_data)
            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=["Periode"], keep="last").sort_values("Periode").reset_index(drop=True)
        else:
            combined_data = new_data.copy()

        # VALIDASI minimal 24 bulan
        if len(combined_data) < 24:
            st.error("❌ Data harus minimal 24 bulan (2 tahun).")
            st.stop()

        combined_data.to_csv(data_filename, index=False, date_format="%Y-%m-%d")

        st.dataframe(combined_data.tail(10))
        st.session_state["data_saved_ok"] = True

        # JIKA berhasil upload data → hapus flag reset agar tombol restore menghilang
        if os.path.exists(RESET_FLAG):
            os.remove(RESET_FLAG)


        st.success("✅ Data sudah tersimpan. Silakan menuju halaman *Analisis* untuk proses training.")

    except Exception as e:
        st.error(f"Gagal memproses file: {e}")

# setelah
# st.success("✅ Data sudah tersimpan. Silakan menuju halaman *Analisis* untuk proses training.")

if st.session_state.get("data_saved_ok", False):
    if st.button("➡️ Lanjut ke Halaman Insights"):
        st.switch_page("pages/analisis.py")

# hanya tampil kalau user habis reset dan belum upload data lagi
if os.path.exists(RESET_FLAG) and not st.session_state.get("data_saved_ok", False):
    if st.button("♻️ Kembalikan Model & Data dari Backup"):
        backup_files = [f for f in os.listdir() if f.endswith("_model_backup.pkl")]
        if backup_files:
            restored_dataset_name = backup_files[0].replace("_model_backup.pkl", "")
            backup_data = f"{restored_dataset_name}_data_backup.csv"
            backup_model = f"{restored_dataset_name}_model_backup.pkl"
            data_file = f"{restored_dataset_name}_data.csv"
            model_file = f"{restored_dataset_name}_model.pkl"

            shutil.copy(backup_model, model_file)
            shutil.copy(backup_data, data_file)

            with open("active_dataset.txt", "w") as f:
                f.write(restored_dataset_name)

            st.session_state["active_dataset"] = restored_dataset_name
            st.success(f"✅ Dataset '{restored_dataset_name}' sudah dikembalikan.")

            time.sleep(1)
            st.switch_page("pages/analisis.py")
        else:
            st.warning("Tidak ada file backup ditemukan.")
