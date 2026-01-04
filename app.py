import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

# --- 2. LOAD MODEL & DATA ---
@st.cache_resource
def load_assets():
    model = joblib.load('gold_model.pkl')
    df = pd.read_csv('gold_model_dataset_2015_2025.csv').dropna()
    return model, df

model, df = load_assets()

# Ambil data terbaru
latest_data = df.iloc[-1]
latest_date = latest_data['Date']

# --- 3. UI: HEADER ---
st.title("💰 Sistem Prediksi Harga Emas")
st.markdown(f"**Status Data:** Referensi pasar terakhir: `{latest_date}`")
st.divider()

# --- 4. UI: SIDEBAR / INPUT ---
st.sidebar.header("Simulasi Indikator Ekonomi")

# Input fitur ekonomi
spx = st.sidebar.number_input("Indeks S&P 500 (SPX)", value=float(latest_data['SPX']), step=10.0)
uso = st.sidebar.number_input("Harga Minyak (USO)", value=float(latest_data['USO']), step=1.0)
slv = st.sidebar.number_input("Harga Perak (SLV)", value=float(latest_data['SLV']), step=0.5)
eur_usd = st.sidebar.number_input("Kurs EUR/USD", value=float(latest_data['EUR/USD']), format="%.4f")

st.sidebar.divider()
# Tambahkan input Kurs Rupiah (Default: 16700)
kurs_idr = st.sidebar.number_input("Atur Kurs USD ke IDR", value=16700, step=100)

# --- 5. LOGIKA PREDIKSI & KONVERSI ---
input_data = np.array([[spx, uso, slv, eur_usd]])
prediction_usd = model.predict(input_data)[0]
prediction_idr = prediction_usd * kurs_idr # Rumus Konversi

# --- 6. UI: DISPLAY HASIL ---
st.subheader("Hasil Analisis Prediksi")
col1, col2 = st.columns(2)

with col1:
    # Tampilkan dalam USD
    st.metric(label="Prediksi Harga (USD)", 
              value=f"${prediction_usd:.2f}", 
              delta=f"{prediction_usd - latest_data['GLD']:.2f} USD")

with col2:
    # Tampilkan dalam IDR (Rupiah)
    # Gunakan pemisah ribuan agar mudah dibaca
    formatted_idr = f"Rp {prediction_idr:,.0f}".replace(",", ".")
    st.metric(label="Konversi ke Rupiah (IDR)", 
              value=formatted_idr)

# --- 7. FOOTER INFORMASI ---
st.info(f"Catatan: Konversi menggunakan kurs Rp {kurs_idr:,}.00 per 1 USD.".replace(",", "."))
st.line_chart(df['GLD'].tail(100))