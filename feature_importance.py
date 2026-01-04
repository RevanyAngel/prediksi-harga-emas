import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
df = pd.read_csv('gold_model_dataset_2015_2025.csv').dropna()
X = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

# 2. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Mendapatkan Feature Importance
importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Fitur': features, 'Kepentingan': importances})
feature_df = feature_df.sort_values(by='Kepentingan', ascending=False)

# 4. Visualisasi
plt.figure(figsize=(10, 6))
sns.barplot(x='Kepentingan', y='Fitur', data=feature_df, palette='viridis')
plt.title('Tingkat Pengaruh Variabel Terhadap Harga Emas', fontsize=14, fontweight='bold')
plt.xlabel('Skor Kepentingan (0 - 1)', fontsize=12)
plt.ylabel('Indikator Ekonomi', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()