import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
try:
    df = pd.read_csv('gold_model_dataset_2015_2025.csv')
    df = df.dropna()

    # 2. Konversi kolom 'Date' menjadi format Tanggal (Datetime)
    # Ini sangat penting agar VS Code tahu urutan waktu yang benar di sumbu X
    df['Date'] = pd.to_datetime(df['Date'])

    # 3. Membuat Visualisasi Line Chart (Time-Series)
    plt.figure(figsize=(12, 6))
    
    # Plotting: sumbu X adalah Date, sumbu Y adalah GLD
    plt.plot(df['Date'], df['GLD'], color='orange', linewidth=1.5, label='Harga GLD')

    # 4. Menambahkan Informasi Label dan Estetika
    plt.title('Tren Harga Emas (GLD) Periode 2015 - 2025', fontsize=14, fontweight='bold')
    plt.xlabel('Tahun', fontsize=12)
    plt.ylabel('Harga GLD (USD)', fontsize=12)
    
    # Menampilkan grid (garis bantu) agar fluktuasi lebih mudah dibaca
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Memberi legenda
    plt.legend()

    # Menampilkan grafik
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: File 'gold_model_dataset_2015_2025.csv' tidak ditemukan.")