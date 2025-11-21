from pmdarima.datasets import load_airpassengers
import pandas as pd
from prophet import Prophet

# Veriyi yükle
df = load_airpassengers


# Prophet formatına uygun isimlendirme
df.columns = ['ds', 'y']

# Tarih formatını datetime objesine çevirme
df['ds'] = pd.to_datetime(df['ds'])

