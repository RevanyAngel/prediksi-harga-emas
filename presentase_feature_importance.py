import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load & Clean Data
df = pd.read_csv('gold_model_dataset_2015_2025.csv').dropna()
X = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

# 2. Split & Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. MENGAMBIL ANGKA FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = X.columns

# 4. MENAMPILKAN HASIL DALAM PERSENTASE
print("=== HASIL FEATURE IMPORTANCE ===")
for name, importance in zip(feature_names, importances):
    percentage = importance * 100
    print(f"{name}: {percentage:.2f}%")