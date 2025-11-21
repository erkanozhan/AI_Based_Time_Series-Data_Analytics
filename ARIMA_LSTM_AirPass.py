# Gerekli kütüphaneleri içe aktarıyoruz.
import pandas as pd  # Veri manipülasyonu ve analizi için temel kütüphane.
import numpy as np  # Sayısal hesaplamalar için temel kütüphane.
import matplotlib.pyplot as plt  # Veri görselleştirme için kullanılır.
from statsmodels.tsa.seasonal import seasonal_decompose  # Zaman serisi bileşenlerini ayrıştırmak için.
from pmdarima.datasets import load_airpassengers  # AirPassengers veri setini yüklemek için.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # ACF ve PACF grafikleri için
from pmdarima import auto_arima # En uygun ARIMA modelini bulmak için
from sklearn.preprocessing import MinMaxScaler # Veriyi belirli bir aralığa ölçeklemek için
from tensorflow.keras.models import Sequential # Katmanları sıralı bir şekilde ekleyebileceğimiz model türü
from tensorflow.keras.layers import LSTM, Dense # LSTM katmanı ve tam bağlantılı katman
from sklearn.metrics import mean_squared_error # Hata metriklerinden Ortalama Kare Hata

# Veriyi bir numpy dizisine dönüştürüp yeniden şekillendiriyoruz.
# Çünkü scaler 2 boyutlu bir dizi bekler.
dataset = data.values.reshape(-1, 1)
dataset = dataset.astype('float32') # Veri tipini float yapıyoruz.

# Veriyi 0-1 aralığına ölçekliyoruz.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

mape_lstm = mean_absolute_percentage_error(testY_inv[0], test_predict[:,0]) * 100
print(f'LSTM Modeli MAPE Değeri: %{mape_lstm:.2f}')