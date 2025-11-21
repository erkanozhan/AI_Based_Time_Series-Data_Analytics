import pandas as pd
from prophet import Prophet

import pandas as pd
from prophet import Prophet

# Veriyi yükle
df = pd.read_csv('data/AirPassengers.csv')

# Prophet formatına uygun isimlendirme
df.columns = ['ds', 'y']

# Tarih formatını datetime objesine çevirme
df['ds'] = pd.to_datetime(df['ds'])

# Modeli başlat
m = Prophet(yearly_seasonality=True, daily_seasonality=False)

# Modeli veriye uydur (Eğitim)
m.fit(df)

# Gelecek 12 ay için boş tarih satırları oluştur
future = m.make_future_dataframe(periods=12, freq='MS') # MS: Month Start

# Tahmin yap
forecast = m.predict(future)

# Sonuçları incele (ds, yhat, yhat_lower, yhat_upper)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
