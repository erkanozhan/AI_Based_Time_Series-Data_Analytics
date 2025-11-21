## XGBoost İle Tahmin
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Gecikme (Lag) özelliği oluşturma
df['lag_1'] = df['Passengers'].shift(1)
df['lag_2'] = df['Passengers'].shift(2)

# Ay bilgisini sayısal özellik olarak ekle
df['month_index'] = df.index.month

# NaN değerleri olan satırları temizle
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

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False)

# Tahmin ve hata hesaplama
y_pred = reg.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f'Test RMSE: {rmse:.2f}')