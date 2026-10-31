# %% [markdown]
# # Bölüm 6 — ACF ve PACF (Python)
# Ders notu: notlar/06_veri_manipulasyonu_gorsellestirme_ve_acf_pacf.md, Bölüm 6.3.7 (ACF) ve 6.4.3 (PACF). Bu dosya iki alt bölümdeki Python kodlarının tamamını içerir;
# R karşılıkları `Codes/R/ch06_manipulasyon_acf.R` dosyasındadır.
# Çalıştırma: depo kök dizininden `python Codes/python/ch06_acf_pacf.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DATA_URL = "https://raw.githubusercontent.com/erkanozhan/AI_Based_Time_Series-Data_Analytics/main/data/"

def veri_yolu(dosya):
    """Veriyi önce yerel data/ klasöründe arar; bulamazsa GitHub'dan okur (Colab için)."""
    for aday in (Path("data") / dosya, Path("../data") / dosya, Path("../../data") / dosya):
        if aday.exists():
            return str(aday)
    return DATA_URL + dosya

# %% [markdown]
# ## 6.3.7 Python ile ACF
# `[20, 22, 21, 23, 24]` serisi için lag-1 otokorelasyonu. Elle hesap: 1 / 10 = 0.1.

# %%
data = np.array([20, 22, 21, 23, 24])   # örnek zaman serisi
acf_values = acf(data, nlags=2)         # lag-0, lag-1, lag-2: [1.0, 0.1, 0.0]
print(f"Lag-1 ACF: {acf_values[1]:.3f}")  # Python 0'dan sayar: [1] lag-1'dir -> 0.100

# %% [markdown]
# ### ACF grafiği (korelogram)
# Python'daki `plot_acf()` bandı varsayılan olarak Bartlett formülüyle çizer: bant gecikme arttıkça genişler.
# R'daki sabit ±1.96/√T bandını görmek için `bartlett_confint=False` yazın.

# %%
seri = pd.read_csv(veri_yolu("AirPassengers.csv"), index_col=0)["Passengers"]
plot_acf(seri, lags=36, title="AirPassengers ACF")    # lags: kaç gecikme çizileceği (R'da lag.max)
plot_acf(seri, lags=36, bartlett_confint=False,
         title="AirPassengers ACF (sabit bant, R ile aynı)")
plt.show()

# %% [markdown]
# ## 6.4.3 Python ile PACF
# Aynı seri için lag-2 kısmi otokorelasyonu. Elle hesap (R ile aynı): (0 − 0.1²) / (1 − 0.1²) ≈ −0.0101.

# %%
pacf_values = pacf(data, nlags=2)       # nlags en fazla gözlem sayısının yarısı: 5 // 2 = 2
print(f"Lag-2 PACF: {pacf_values[2]:.3f}")  # varsayılan method="ywadjusted" -> -0.016

# R ile aynı sonuç için otokovaryansları T ile bölen yöntem:
print(f"Lag-2 PACF (ywm): {pacf(data, nlags=2, method='ywm')[2]:.3f}")  # -> -0.010

# %% [markdown]
# ### PACF grafiği
# `plot_pacf()` varsayılan olarak `method="ywm"` kullanır ve sabit ±1.96/√T bandını çizer; değerler R'ınkiyle aynıdır.

# %%
plot_pacf(seri, lags=36, title="AirPassengers PACF")
plt.show()
