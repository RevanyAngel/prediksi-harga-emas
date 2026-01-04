import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load & Preprocessing
df = pd.read_csv('gold_model_dataset_2015_2025.csv').dropna()
X = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

# 2. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Prediksi
y_pred = model.predict(X_test)

# 4. Visualisasi Perbandingan
plt.figure(figsize=(12, 6))
# Plot harga asli
plt.plot(y_test.values, color='blue', label='Harga Asli (Actual)', alpha=0.6)
# Plot harga prediksi
plt.plot(y_pred, color='red', label='Harga Prediksi (Predicted)', alpha=0.6)

plt.title('Hasil Prediksi Random Forest: Harga Asli vs Prediksi (2015-2025)', fontsize=14, fontweight='bold')
plt.xlabel('Urutan Data Uji', fontsize=12)
plt.ylabel('Harga GLD (USD)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Simpan model ke dalam file bernama 'gold_model.pkl'
joblib.dump(model, 'gold_model.pkl')
print("Model berhasil disimpan!")