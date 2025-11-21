import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Veriyi yükle
df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Gecikme (Lag) özelliği oluşturma
df['lag_1'] = df['Passengers'].shift(1)
df['lag_2'] = df['Passengers'].shift(2)

# Ay bilgisini sayısal özellik olarak ekle
df['month_index'] = df.index.month

# NaN (boş) değerleri at (Lag özellikleri oluştururken ilk satırlar boş kalır)
df = df.dropna()

# X: Girdiler, y: Hedef
X = df[['lag_1', 'lag_2', 'month_index']]
y = df['Passengers']

# Son 12 ayı test olarak ayır
split_point = len(df) - 12
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# XGBoost Regresyon modeli
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)

# Modeli eğit
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False)

# Tahmin yapma
y_pred = reg.predict(X_test)

# Hata hesaplama (RMSE ve MAE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE Değeri: {rmse:.2f}")
print(f"MAE Değeri: {mae:.2f}")

# Sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Gerçek Değerler')
plt.plot(y_test.index, y_pred, label='Tahminler', linestyle='--', color='red')
plt.title('XGBoost ile AirPassengers Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()