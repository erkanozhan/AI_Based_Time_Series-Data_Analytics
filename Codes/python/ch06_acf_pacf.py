# %% [markdown]
# # Bölüm 6 — ACF ve PACF (Python)
# Ders notu: notlar/06_veri_manipulasyonu_gorsellestirme_ve_acf_pacf.md, Bölüm 6.4.3 (ACF) ve 6.5.3 (PACF). Bu dosya iki alt bölümdeki Python kodlarının tamamını içerir;
# R karşılıkları `Codes/R/ch06_manipulasyon_acf.R` dosyasındadır.
# Çalıştırma: depo kök dizininden `python Codes/python/ch06_acf_pacf.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.

# %%
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
# ## 6.4.3 Python ile ACF grafiği (korelogram)
# Python'daki `plot_acf()` bandı varsayılan olarak Bartlett formülüyle çizer: bant gecikme arttıkça genişler.
# R'daki sabit ±1.96/√T bandını görmek için `bartlett_confint=False` yazın.

# %%
seri = pd.read_csv(veri_yolu("AirPassengers.csv"), index_col=0)["Passengers"]
print(acf(seri, nlags=12)[[1, 12]].round(3))           # lag-1 ve lag-12 (R ile aynı formül): yaklaşık 0.948 ve 0.760
plot_acf(seri, lags=36, title="AirPassengers ACF")    # lags: kaç gecikme çizileceği (R'da lag.max)
plot_acf(seri, lags=36, bartlett_confint=False,
         title="AirPassengers ACF (sabit bant, R ile aynı)")
plt.show()

# %% [markdown]
# ## 6.5.3 Python ile PACF
# `pacf()` varsayılan olarak `method="ywadjusted"` (T - h bölenli) kullanır; R ile aynı değerler için `method="ywm"` (T bölenli) yazılır.
# Python 0'dan sayar ve sonuç lag-0'ı içerir: [1] lag-1, [2] lag-2'dir.

# %%
print("Varsayılan (ywadjusted):", pacf(seri, nlags=2).round(4))
print("R ile aynı (ywm):       ", pacf(seri, nlags=2, method="ywm").round(4))

# %% [markdown]
# ### PACF grafiği
# `plot_pacf()` varsayılan olarak `method="ywm"` kullanır ve sabit ±1.96/√T bandını çizer; değerler R'ınkiyle aynıdır.

# %%
plot_pacf(seri, lags=36, title="AirPassengers PACF")
plt.show()
