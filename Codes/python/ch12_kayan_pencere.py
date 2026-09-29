# %% [markdown]
# # Bölüm 12 — Zaman Serisini Denetimli Öğrenmeye Dönüştürme: Kayan Pencere ve Özellikler
# Ders notu: Course_notes.md, Bölüm 12.1. Bu dosya bölümdeki Python kodlarının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch12_kayan_pencere.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/erkanozhan/AI_Based_Time_Series-Data_Analytics/main/data/"

def veri_yolu(dosya):
    """Veriyi önce yerel data/ klasöründe arar; bulamazsa GitHub'dan okur (Colab için)."""
    for aday in (Path("data") / dosya, Path("../data") / dosya, Path("../../data") / dosya):
        if aday.exists():
            return str(aday)
    return DATA_URL + dosya

# Veriyi yükleyelim: tarih sütununu indeks yapıyoruz
df = pd.read_csv(veri_yolu('AirPassengers.csv'), parse_dates=['Month'], index_col='Month')
seri = df['Passengers']

# %% [markdown]
# ## 12.1.1 Kayan pencere (sliding window)
# Son `p` gözlem girdi (`X`), hemen sonraki gözlem hedef (`y`) olur.

# %%
# 1) Kayan pencere: son p gözlem -> bir sonraki gözlem
def kayan_pencere(dizi, p):
    X, y = [], []
    for i in range(len(dizi) - p):
        X.append(dizi[i:i + p])   # girdi: p uzunluğunda pencere
        y.append(dizi[i + p])     # hedef: pencereden hemen sonraki değer
    return np.array(X), np.array(y)

X, y = kayan_pencere(seri.values, p=3)
print(X.shape, y.shape)   # 144 gözlemden 144 - 3 = 141 örnek
print(X[:3], y[:3])

# %% [markdown]
# ## 12.1.2 Gecikme ve takvim özellikleri
# `shift(k)` ile gecikmeler, `shift(1).rolling(...)` ile sızıntısız hareketli ortalama, indeksten takvim özellikleri.

# %%
# 2) Aynı fikir pandas ile: gecikme ve takvim özellikleri
tablo = pd.DataFrame({'y': seri})
for k in [1, 2, 3, 12]:                     # x_{t-1}, x_{t-2}, x_{t-3} ve geçen yılın aynı ayı
    tablo[f'lag_{k}'] = seri.shift(k)
# Hareketli ortalama yalnızca GEÇMİŞ değerlerden hesaplanır: shift(1) sızıntıyı önler
tablo['ort_3'] = seri.shift(1).rolling(window=3).mean()
tablo['ay'] = tablo.index.month             # takvim özelliği: 1-12
tablo['ceyrek'] = tablo.index.quarter       # takvim özelliği: 1-4
tablo = tablo.dropna()                      # geçmişi eksik ilk 12 satır atılır

print(tablo.head(3))

# %% [markdown]
# ## Zamansal ayrım (bkz. 12.1.5 veri sızıntısı)
# Son 12 ay test seti; satırlar karıştırılmaz.

# %%
# 3) Zamansal ayrım: son 12 ay test, veri KARIŞTIRILMAZ
X_train, X_test = tablo.drop(columns='y').iloc[:-12], tablo.drop(columns='y').iloc[-12:]
y_train, y_test = tablo['y'].iloc[:-12], tablo['y'].iloc[-12:]
print(X_train.shape, X_test.shape)
