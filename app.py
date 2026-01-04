import streamlit as st
import joblib
import numpy as np

# 1. Load model
model = joblib.load('gold_model.pkl')

# 2. Judul Aplikasi
st.title("Aplikasi Prediksi Harga Emas (GLD)")
st.write("Masukkan indikator ekonomi di bawah ini untuk memprediksi harga emas.")

# 3. Input User (Sidebar atau kolom utama)
spx = st.number_input("Indeks S&P 500 (SPX)", value=3000.0)
uso = st.number_input("Harga Minyak (USO)", value=70.0)
slv = st.number_input("Harga Perak (SLV)", value=15.0)
eur_usd = st.number_input("Kurs EUR/USD", value=1.1)

# 4. Tombol Prediksi
if st.button("Prediksi Sekarang"):
    # Gabungkan input menjadi array 2D
    input_data = np.array([[spx, uso, slv, eur_usd]])
    
    # Lakukan prediksi
    prediction = model.predict(input_data)
    
    # Tampilkan hasil
    st.success(f"Estimasi Harga Emas (GLD): ${prediction[0]:.2f}")

    