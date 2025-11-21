## XGBoost İle Tahmin
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Gecikme (Lag) özelliği oluşturma
# Bir önceki ayın (t-1) verisini yan sütuna taşıyoruz
df['lag_1'] = df['Passengers'].shift(1)
df['lag_2'] = df['Passengers'].shift(2) # İki ay öncesi

# Ayrıca Ay bilgisini sayısal bir özellik olarak ekleyelim (Ocak=1, Şubat=2...)
df['month_index'] = df.index.month

# X: Girdiler (t-1, t-2 ve ay bilgisi), y: Hedef (t anındaki yolcu sayısı)
X = df[['lag_1', 'lag_2', 'month_index']]
y = df['Passengers']

# Son 12 ayı test olarak ayıralım
split_point = len(df) - 12
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# XGBoost Regresyon modelini çağırma
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, # İyileşme durursa eğitimi kes
        verbose=False)