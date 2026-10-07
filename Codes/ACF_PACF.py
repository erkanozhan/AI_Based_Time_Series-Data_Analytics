# =============================================================================
# Bölüm 6 — ACF ve PACF (Python)
# Ders notu: Course_notes.md, Bölüm 6.3.7 (ACF) ve 6.4.3 (PACF)
#
# Çalıştırma: depo kök dizininden  python Codes/ACF_PACF.py
# Gerekli paketler: numpy, pandas, matplotlib, statsmodels
# =============================================================================

from statsmodels.tsa.stattools import acf  # ACF fonksiyonunu içeri aktar
import numpy as np # Numpy kütüphanesini içeri aktar

# ---- 6.3.7 Python ile ACF ----
data = np.array([20, 22, 21, 23, 24]) # Örnek bir zaman serisi verisi oluştur
acf_values = acf(data, nlags=2) # 2 gecikmeye kadar ACF değerlerini hesapla
print(f"Lag-1 ACF: {acf_values[1]:.3f}") # 1. gecikmedeki (lag-1) ACF değerini yazdır

# ---- 6.4.3 Python ile PACF ----
from statsmodels.tsa.stattools import pacf  # PACF fonksiyonunu içeri aktar

pacf_values = pacf(data, nlags=2)  # nlags, gözlem sayısının yarısını aşamaz
print(f"Lag-2 PACF: {pacf_values[2]:.3f}")  # varsayılan method="ywadjusted": -0.016
# R ile aynı sonuç (-0.010) için otokovaryansları T ile bölen yöntem:
print(f"Lag-2 PACF (ywm): {pacf(data, nlags=2, method='ywm')[2]:.3f}")

# ---- 6.3.7 ACF grafiğini çizmek (ve 6.4.3 PACF grafiği) ----
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

seri = pd.read_csv("data/AirPassengers.csv", index_col=0)["Passengers"]
plot_acf(seri, lags=36, title="AirPassengers ACF")  # lags: kaç gecikme çizileceği
# R ile aynı sabit bandı görmek için: plot_acf(seri, lags=36, bartlett_confint=False)
plot_pacf(seri, lags=36, title="AirPassengers PACF")  # varsayılan method="ywm" (R ile aynı)
plt.show()
