import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load & Clean
df = pd.read_csv('gold_model_dataset_2015_2025.csv')
print(f"Jumlah awal: {len(df)}")

df = df.dropna()
print(f"Jumlah setelah dropna: {len(df)}")

# 2. Pisahkan Fitur dan Target
X = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Hasil Splitting ---")
print(f"Jumlah Data Latih (X_train): {len(X_train)}")
print(f"Jumlah Data Uji (X_test): {len(X_test)}")