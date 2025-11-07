from statsmodels.tsa.stattools import acf # ACF fonksiyonunu içeri aktar
import numpy as np # Numpy kütüphanesini içeri aktar

data = np.array([20, 22, 21, 23, 24]) # Örnek bir zaman serisi verisi oluştur
acf_values = acf(data, nlags=2) # 2 gecikmeye kadar ACF değerlerini hesapla
print(f"Lag-1 ACF: {acf_values[1]:.3f}") # 1. gecikmedeki (lag-1) ACF değerini yazdır