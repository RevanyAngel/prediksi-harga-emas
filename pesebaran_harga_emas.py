import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
# Pastikan file CSV berada di folder yang sama dengan file script ini
try:
    df = pd.read_csv('gold_model_dataset_2015_2025.csv')

    # 2. Pembersihan Data (Sesuai Bab 3.4)
    # Menghapus baris kosong yang biasanya ada di baris pertama/awal dataset
    df = df.dropna()

    # 3. Pengaturan Style Visualisasi
    sns.set_theme(style="whitegrid")

    # 4. Membuat Visualisasi Distribusi GLD
    plt.figure(figsize=(10, 6))
    # bins=30 untuk membagi data menjadi 30 kotak agar kepadatan terlihat jelas
    # kde=True untuk menambahkan garis tren kurva (Kernel Density Estimate)
    sns.histplot(df['GLD'], kde=True, color='gold', bins=30)

    # Menambahkan Informasi Label dan Judul
    plt.title('Distribusi Harga Emas (GLD) Periode 2015 - 2025', fontsize=14, fontweight='bold')
    plt.xlabel('Harga GLD (USD)', fontsize=12)
    plt.ylabel('Frekuensi Munculnya Harga', fontsize=12)

    # Menampilkan grafik
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: File 'gold_model_dataset_2015_2025.csv' tidak ditemukan.")