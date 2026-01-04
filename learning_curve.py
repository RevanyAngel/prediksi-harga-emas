import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- 1. LOAD & CLEAN DATA ---
try:
    df = pd.read_csv('gold_model_dataset_2015_2025.csv').dropna()
    X = df.drop(['Date', 'GLD'], axis=1)
    y = df['GLD']

    # --- 2. KONFIGURASI LEARNING CURVE ---
    # n_jobs=-1 menggunakan semua core prosesor agar lebih cepat
    # cv=5 menggunakan 5-Fold Cross Validation
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestRegressor(n_estimators=100, random_state=42), 
        X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='neg_mean_absolute_error'
    )

    # Mengubah nilai skor menjadi positif (karena scoring sklearn menghasilkan nilai negatif)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    # --- 3. VISUALISASI ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error (MAE)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Error (MAE)")

    # Menambahkan detail grafik
    plt.title('Learning Curve: Analisis Performa Model Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Jumlah Data Latih (Training Examples)', fontsize=12)
    plt.ylabel('Error (Mean Absolute Error)', fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: File 'gold_model_dataset_2015_2025.csv' tidak ditemukan.")