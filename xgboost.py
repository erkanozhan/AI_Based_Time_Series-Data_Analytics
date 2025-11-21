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