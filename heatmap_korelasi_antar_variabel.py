import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
try:
    df = pd.read_csv('gold_model_dataset_2015_2025.csv')
    df = df.dropna()

    # 2. Menghitung Korelasi
    # Kita harus menghapus kolom 'Date' karena korelasi hanya bisa dihitung untuk angka
    correlation = df.drop('Date', axis=1).corr()

    # 3. Membuat Visualisasi Heatmap
    plt.figure(figsize=(10, 8))
    
    # annot=True: Menampilkan angka korelasi di dalam kotak
    # cmap='RdYlGn': Warna (Merah untuk negatif, Kuning netral, Hijau untuk positif)
    # fmt='.2f': Menampilkan 2 angka di belakang koma
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)

    # Menambahkan Judul
    plt.title('Heatmap Korelasi Antar Variabel Ekonomi (2015 - 2025)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: File 'gold_model_dataset_2015_2025.csv' tidak ditemukan.")