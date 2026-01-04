import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
try:
    df = pd.read_csv('gold_model_dataset_2015_2025.csv')
    df = df.dropna()

    # 2. Membuat Scatter Plot dengan Regression Line
    plt.figure(figsize=(10, 6))
    
    # x = SPX (Pemicu), y = GLD (Target)
    # scatter_kws={'alpha':0.5}: Membuat titik agak transparan agar penumpukan data terlihat
    # line_kws={'color':'red'}: Menambahkan garis tren berwarna merah
    sns.regplot(x='SPX', y='GLD', data=df, 
                scatter_kws={'alpha':0.4, 'color':'blue'}, 
                line_kws={'color':'red'})

    # 3. Menambahkan Label dan Judul
    plt.title('Analisis Hubungan S&P 500 (SPX) terhadap Harga Emas (GLD)', fontsize=14, fontweight='bold')
    plt.xlabel('Indeks S&P 500 (SPX)', fontsize=12)
    plt.ylabel('Harga Emas (GLD) dalam USD', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.3)

    # Menampilkan grafik
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: File 'gold_model_dataset_2015_2025.csv' tidak ditemukan.")