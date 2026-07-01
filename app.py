import streamlit as st
import numpy as np
import joblib

# ==========================================
# Konfigurasi Halaman
# ==========================================
st.set_page_config(
    page_title="Simulator Risiko Mesin",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Simulator Risiko Kerusakan Mesin")
st.write(
    "Aplikasi ini memprediksi **Skor Risiko Kerusakan Mesin** "
    "berdasarkan suhu dan getaran mesin."
)

# ==========================================
# Model Versioning
# ==========================================
MODEL_VERSION = "v1"

MODEL_PATH = f"model_risiko_{MODEL_VERSION}.joblib"
SCALER_PATH = f"scaler_risiko_{MODEL_VERSION}.joblib"

# Load model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==========================================
# Input User
# ==========================================

st.header("Input Data Sensor")

suhu_input = st.number_input(
    "🌡 Suhu Mesin (°C)",
    min_value=0.0,
    max_value=300.0,
    value=85.0,
    step=1.0
)

getaran_input = st.number_input(
    "📈 Getaran Mesin",
    min_value=0.0,
    max_value=50.0,
    value=7.0,
    step=0.1
)

# ==========================================
# Monitoring Drift
# ==========================================

st.subheader("Monitoring Data")

if suhu_input > 120 or suhu_input < 10:
    st.warning(
        "⚠ Input suhu berada di luar jangkauan data latihan. "
        "Hasil simulasi mungkin kurang akurat."
    )

if getaran_input > 15:
    st.warning(
        "⚠ Nilai getaran cukup tinggi dibandingkan data latihan."
    )

# ==========================================
# Prediksi
# ==========================================

if st.button("Prediksi Risiko"):

    data_baru = np.array([[suhu_input, getaran_input]])

    # Konsistensi preprocessing
    data_scaled = scaler.transform(data_baru)

    hasil = model.predict(data_scaled)[0]

    st.success(f"Skor Risiko : **{hasil:.2f}**")

    # Interpretasi sederhana
    if hasil < 30:
        st.success("🟢 Risiko Rendah")
    elif hasil < 70:
        st.warning("🟡 Risiko Sedang")
    else:
        st.error("🔴 Risiko Tinggi")