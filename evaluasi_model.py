import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# --- 1. LOAD & CLEAN DATA ---
df = pd.read_csv('gold_model_dataset_2015_2025.csv')
df = df.dropna()

X = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

# --- 2. SPLIT & TRAIN ---
# Kita harus mendefinisikan y_test di sini agar tidak NameError
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 3. PREDIKSI ---
# Kita harus membuat y_pred di sini
y_pred = model.predict(X_test)

# --- 4. HITUNG METRIK ---
r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("=== HASIL EVALUASI MODEL ===")
print(f"R-Squared (R2) : {r2:.4f}")
print(f"Mean Abs Error : {mae:.2f} USD")
print(f"RMSE           : {rmse:.2f} USD")
print("============================")

# --- 5. VISUALISASI EVALUASI ---
plt.figure(figsize=(15, 5))

# Plot 1: Scatter Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='darkblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted\n(R2 = {r2:.4f})')
plt.xlabel('Harga Asli (USD)')
plt.ylabel('Harga Prediksi (USD)')

# Plot 2: Residual Plot (Error)
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot\n(Analisis Error)')
plt.xlabel('Prediksi')
plt.ylabel('Selisih (Actual-Pred)')

# Plot 3: Distribusi Error
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True, color='green')
plt.title(f'Distribusi Error\n(MAE = {mae:.2f})')
plt.xlabel('Besar Error (USD)')

plt.tight_layout()
plt.show()