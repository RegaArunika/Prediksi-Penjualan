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
st.subheader("📂 Input Data Baru & Latih Model")
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload data penjualan (CSV/XLSX, kolom: Periode & Pemasukan)", type=["csv", "xlsx"])

contoh_data = pd.DataFrame({
    "Periode": pd.date_range("2025-01-01", periods=4, freq="MS").strftime("%Y-%m-%d"),
    "Pemasukan": [12000000, 13500000, 15000000, 14500000]
})
csv_buffer = io.StringIO()
contoh_data.to_csv(csv_buffer, index=False)
st.download_button("⬇️ Download Contoh Data (CSV)", csv_buffer.getvalue(), "contoh_data.csv", "text/csv")


active_dataset = None
if os.path.exists("active_dataset.txt"):
    with open("active_dataset.txt") as f:
        active_dataset = f.read().strip()


sarima_model = None
if active_dataset and os.path.exists(f"{active_dataset}_model.pkl"):
    with open(f"{active_dataset}_model.pkl", "rb") as f:
        sarima_model = pickle.load(f)


# ============================================================
# Proses Upload
# ============================================================
if uploaded_file:
    current_dataset_name = active_dataset
    if current_dataset_name is None:
        current_dataset_name = Path(uploaded_file.name).stem
        with open("active_dataset.txt", "w") as f:
            f.write(current_dataset_name)
        active_dataset = current_dataset_name

    data_filename = f"{current_dataset_name}_data.csv"

    try:
        # Baca file upload (parse dates jika csv)
        if uploaded_file.name.endswith(".csv"):
            new_data = pd.read_csv(uploaded_file)
        else:
            new_data = pd.read_excel(uploaded_file)

        if "Periode" not in new_data.columns or "Pemasukan" not in new_data.columns:
            st.error("❌ File harus memiliki kolom 'Periode' dan 'Pemasukan'.")
            st.stop()

        # Preprocess & normalisasi
        new_data = preprocess_period_column(new_data)

        if os.path.exists(data_filename):
            old_data = pd.read_csv(data_filename)
            old_data = preprocess_period_column(old_data)
            st.info(f"Dataset lama: {len(old_data)} baris (setelah normalisasi)")

            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            before = len(combined_data)
            # Hapus duplikat berdasar Periode (setiap periode hanya 1 baris)
            combined_data = combined_data.drop_duplicates(subset=["Periode"], keep="last").sort_values("Periode").reset_index(drop=True)
            after = len(combined_data)
            removed = before - after
            st.success(f"✅ Penggabungan selesai. {removed} baris duplikat dihapus. Total sekarang: {after} baris.")
        else:
            combined_data = new_data.copy()
            st.success(f"✅ Dataset '{current_dataset_name}' berhasil diinputkan ({len(combined_data)} baris).")

        # Simpan dengan format tanggal konsisten
        combined_data.to_csv(data_filename, index=False, date_format="%Y-%m-%d")
        st.info("Data terbaru yang digunakan:")
        st.dataframe(combined_data.tail(10))

    except Exception as e:
        st.error(f"Gagal memproses file: {e}")


# ============================================================
# Tombol Train / Retrain Model (DIPINDAHKAN KE SINI)
# ============================================================
# Blok ini akan muncul di bawah uploader file jika sebuah dataset sudah aktif (tersimpan).
if active_dataset and os.path.exists(f"{active_dataset}_data.csv"):
    data_filename = f"{active_dataset}_data.csv"
    model_filename = f"{active_dataset}_model.pkl"

    sales_data = pd.read_csv(data_filename)
    sales_data = preprocess_period_column(sales_data)
    last_period = sales_data["Periode"].iloc[-1].strftime("%B %Y")
    
    # Notifikasi status data dipindahkan ke sini
    st.info(f"📅 Data terakhir: **{last_period}**")

    train_button_label = "🚀 Train Model Baru" if sarima_model is None else "🔁 Retrain Model"
    if st.button(train_button_label):
        with st.spinner("Sedang melatih model... 🧠"):
            try:
                if len(sales_data) < 24:
                    st.warning("⚠️ Jumlah data disarankan minimal 24 bulan untuk hasil optimal.")

                y = sales_data["Pemasukan"]
                y_log = np.log1p(y)
                model = sm.tsa.statespace.SARIMAX(
                    y_log, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False)
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)
                sarima_model = model
                st.success(f"✅ Model untuk '{active_dataset}' berhasil dilatih dengan data hingga {last_period}!")
            except Exception as e:
                st.error(f"Gagal melatih model: {e}")
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

# ============================================================
# Reset & Restore
# ============================================================
st.subheader("⚙️ Pengaturan Model")
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

col_reset, col_restore = st.columns(2)

with col_reset:
    if st.button("🧹 Reset Model & Data"):
        try:
            if active_dataset:
                data_file = f"{active_dataset}_data.csv"
                model_file = f"{active_dataset}_model.pkl"
                backup_data = f"{active_dataset}_data_backup.csv"
                backup_model = f"{active_dataset}_model_backup.pkl"

                if os.path.exists(model_file): shutil.copy(model_file, backup_model)
                if os.path.exists(data_file): shutil.copy(data_file, backup_data)
                for f in [model_file, data_file, "active_dataset.txt"]:
                    if os.path.exists(f): os.remove(f)

                st.warning(f"⚠️ Model dan data '{active_dataset}' telah direset. Silakan refresh halaman.")
            else:
                st.warning("⚠️ Tidak ada dataset aktif untuk direset.")
        except Exception as e:
            st.error(f"Gagal mereset: {e}")

with col_restore:
    if st.button("♻️ Kembalikan Model Setelah Reset"):
        try:
            restored_dataset_name = active_dataset
            if restored_dataset_name is None:
                backup_files = [f for f in os.listdir() if f.endswith("_model_backup.pkl")]
                if backup_files: restored_dataset_name = backup_files[0].replace("_model_backup.pkl", "")

            if restored_dataset_name:
                data_file = f"{restored_dataset_name}_data.csv"
                model_file = f"{restored_dataset_name}_model.pkl"
                backup_data = f"{restored_dataset_name}_data_backup.csv"
                backup_model = f"{restored_dataset_name}_model_backup.pkl"
                if os.path.exists(backup_model) and os.path.exists(backup_data):
                    shutil.copy(backup_model, model_file)
                    shutil.copy(backup_data, data_file)
                    with open("active_dataset.txt", "w") as f: f.write(restored_dataset_name)
                    st.success(f"✅ Model dan data '{restored_dataset_name}' berhasil dikembalikan! Silakan refresh halaman.")
                else:
                    st.warning("⚠️ File backup tidak ditemukan.")
            else:
                st.warning("⚠️ Tidak ada dataset aktif atau backup yang dapat dikembalikan.")
        except Exception as e:
            st.error(f"Gagal mengembalikan: {e}")
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

# ============================================================
# Prediksi & Visualisasi
# ============================================================
if sarima_model is not None and active_dataset and os.path.exists(f"{active_dataset}_data.csv"):
    st.subheader("📈 Prediksi & Visualisasi")
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    n_periods = st.slider("Pilih jumlah bulan ke depan untuk prediksi:", 1, 24, 6)

    # --- Persiapan Data untuk Visualisasi ---
    hist_data = pd.read_csv(f"{active_dataset}_data.csv")
    hist_data = preprocess_period_column(hist_data)
    hist_data["Tipe"] = "Aktual"

    forecast_res = sarima_model.get_forecast(steps=n_periods)
    forecast_mean = np.expm1(forecast_res.predicted_mean)
    conf_int_exp = np.expm1(forecast_res.conf_int())

    forecast_df = pd.DataFrame({
        "Periode": pd.date_range(hist_data["Periode"].iloc[-1] + pd.DateOffset(months=1), periods=n_periods, freq="MS"),
        "Pemasukan": forecast_mean,
        "Tipe": "Prediksi"
    })
    
    combined_vis = pd.concat([hist_data, forecast_df], ignore_index=True)
    
    st.write("Tabel Hasil Prediksi:")
    display_df = forecast_df.copy()
    display_df["Periode"] = display_df["Periode"].dt.strftime("%B %Y")
    display_df["Pemasukan (Rp)"] = display_df["Pemasukan"].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
    st.dataframe(display_df[["Periode", "Pemasukan (Rp)"]])

    # --- Buat dan Tampilkan Semua Grafik ---
    st.write("---")
    fig_line = px.line(combined_vis, x="Periode", y="Pemasukan", color="Tipe", markers=True, 
                       title=f"📈 Aktual vs Prediksi Pemasukan — Dataset: {active_dataset}",
                       color_discrete_map={"Aktual": "#3498db", "Prediksi": "#e74c3c"})
    fig_line.update_layout(legend_title_text="Jenis Data", yaxis_title="Pemasukan (Rp)", xaxis_title="Periode")
    st.plotly_chart(fig_line, use_container_width=True)

    st.write("---")
    fig_ci = px.line(hist_data, x="Periode", y="Pemasukan", title="🎯 Prediksi Pemasukan dengan Rentang Keyakinan (Confidence Interval)")
    fig_ci.data[0].name = 'Aktual'
    fig_ci.data[0].showlegend = True
    fig_ci.add_scatter(x=forecast_df["Periode"], y=forecast_df["Pemasukan"], mode="lines", name="Prediksi", line=dict(color="#e74c3c"))
    fig_ci.add_scatter(x=forecast_df["Periode"], y=conf_int_exp.iloc[:, 1], mode="lines", line=dict(dash="dash", color="green"), name="Batas Atas CI")
    fig_ci.add_scatter(x=forecast_df["Periode"], y=conf_int_exp.iloc[:, 0], mode="lines", line=dict(dash="dash", color="yellow"), name="Batas Bawah CI")
    fig_ci.update_layout(yaxis_title="Pemasukan (Rp)", xaxis_title="Periode", legend_title_text="Keterangan")
    st.plotly_chart(fig_ci, use_container_width=True)

    st.write("---")
    tail_periods = st.slider("Tampilkan N bulan terakhir pada Bar Chart:", 6, 36, 12)
    bar_data = combined_vis.tail(tail_periods)
    
    fig_bar = px.bar(bar_data, x="Periode", y="Pemasukan", color="Tipe",
                     barmode="group", title=f"📊 Perbandingan Pemasukan Bulanan (Aktual vs Prediksi) — {tail_periods} Bulan Terakhir",
                     labels={"Pemasukan": "Pemasukan (Rp)"},
                     color_discrete_map={"Aktual": "#3498db", "Prediksi": "#e74c3c"})
    fig_bar.update_layout(xaxis_title="Periode", yaxis_title="Pemasukan (Rp)", legend_title_text="Jenis Data")
    st.plotly_chart(fig_bar, use_container_width=True)

     # ============================================================
    # Export PowerPoint (Sendiri)
    # ============================================================
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    st.subheader("💾 Export Visualisasi")

    if st.button("📤 Export Visualisasi ke PowerPoint"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmpfile:
                prs = Presentation()
                title_slide_layout = prs.slide_layouts[0]
                slide = prs.slides.add_slide(title_slide_layout)
                slide.shapes.title.text = "Laporan Prediksi Pemasukan"
                slide.placeholders[1].text = (
                    f"Dataset: {active_dataset}\n"
                    f"Periode Data Aktual: {hist_data['Periode'].min().strftime('%B %Y')} - {hist_data['Periode'].max().strftime('%B %Y')}\n"
                    f"Periode Prediksi: {n_periods} bulan ke depan"
                )

                for fig_obj, title in [
                    (fig_line, "Grafik Aktual vs Prediksi"),
                    (fig_ci, "Grafik Rentang Keyakinan Prediksi"),
                    (fig_bar, "Grafik Perbandingan Bulanan")
                ]:
                    slide = prs.slides.add_slide(prs.slide_layouts[5])
                    slide.shapes.title.text = title
                    # Gunakan method write_html dan screenshot alternatif (tanpa Kaleido)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as html_tmp:
                fig_obj.write_html(html_tmp.name)
                import plotly.io as pio
                try:
                    # Coba gunakan kaleido jika ada
                    img_bytes = pio.to_image(fig_obj, format="png", width=960, height=540)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
                        img_tmp.write(img_bytes)
                        img_path = img_tmp.name
                except Exception:
                    # Jika kaleido gagal, buat placeholder
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.new("RGB", (960, 540), color=(245, 245, 245))
                    d = ImageDraw.Draw(img)
                    d.text(
                        (50, 250),
                        f"Tampilan '{title}'\n(Kaleido tidak tersedia)",
                        fill=(0, 0, 0)
                    )
                    img_path = html_tmp.name.replace(".html", ".png")
                    img.save(img_path)

            # Tambahkan gambar (hasil asli atau placeholder) ke slide PowerPoint
            slide.shapes.add_picture(img_path, Inches(0.5), Inches(1.5), width=Inches(9))

            # Bersihkan file sementara
            os.remove(img_path)
            os.remove(html_tmp.name)


            with open(tmpfile.name, "rb") as f:
                ppt_bytes = f.read()

            st.download_button(
                label="⬇️ Download Laporan PowerPoint",
                data=ppt_bytes,
                file_name=f"laporan_prediksi_{active_dataset}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
            os.remove(tmpfile.name)

        except Exception as e:
            st.error(f"Gagal membuat PowerPoint: {e}")

# ============================================================
# Inisialisasi Ulang Sistem (Auto-reload dengan st.rerun)
# ============================================================
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
st.subheader("🧨 Inisialisasi Ulang Sistem")

st.markdown("""
Gunakan tombol di bawah ini untuk menghapus **seluruh data, model, backup, dan file aktif**.  
Aplikasi akan kembali ke kondisi awal secara otomatis setelah proses selesai.
""")

if st.button("🔄 Inisialisasi Ulang Sistem"):
    try:
        # Hapus file di direktori kerja (kecuali logo)
        for file in os.listdir():
            if file.startswith("Logo"):
                continue
            if file.endswith((".pkl", ".csv", ".txt", ".backup.pkl", ".backup.csv")):
                try:
                    os.remove(file)
                except:
                    pass

        # Hapus file di folder temporary
        tmp_dir = tempfile.gettempdir()
        for f in os.listdir(tmp_dir):
            if f.startswith("tmp") and f.endswith((".pptx", ".png")):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except:
                    pass

        # Hapus session state agar data UI ikut ter-reset
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Pesan sukses sebelum reload
        st.success("✅ Semua data, model, dan file sementara telah dihapus.")
        st.info("Memuat ulang aplikasi...")

        # 🔁 Auto refresh / reload halaman
        st.rerun()

    except Exception as e:
        st.error(f"Gagal melakukan reset total: {e}")
