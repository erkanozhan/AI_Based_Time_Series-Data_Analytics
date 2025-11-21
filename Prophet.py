from pmdarima.datasets import load_airpassengers
import pandas as pd
from prophet import Prophet

import pandas as pd
from prophet import Prophet

# Veriyi yükle
df = pd.read_csv('AirPassengers.csv')

# Prophet formatına uygun isimlendirme
df.columns = ['ds', 'y']

# Tarih formatını datetime objesine çevirme
df['ds'] = pd.to_datetime(df['ds'])

