# %% [markdown]
# # Bölüm 7 — Python ile SARIMA: `AirPassengers`
# Ders notu: Course_notes.md, Bölüm 7.7 (Python ile Aynı Analiz). Bu dosya bölümdeki Python kodlarının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch07_sarima_airpassengers.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.

# %% [markdown]
# ## 7.7.1 Veri setinin yüklenmesi ve hazırlanması

# %%
# COLAB: pip install -q pmdarima
# Gerekli kütüphaneleri içe aktarıyoruz.
# Kurulum: pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn
import pandas as pd  # Veri manipülasyonu ve analizi için temel kütüphane.
import numpy as np  # Sayısal hesaplamalar için temel kütüphane.
import matplotlib.pyplot as plt  # Veri görselleştirme için kullanılır.
from pmdarima.datasets import load_airpassengers  # AirPassengers veri setini yüklemek için.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ACF ve PACF grafikleri için
from statsmodels.tsa.stattools import adfuller, kpss  # ADF ve KPSS durağanlık testleri için
from pmdarima import auto_arima  # En uygun (S)ARIMA modelini otomatik bulmak için
from sklearn.metrics import mean_squared_error  # Ortalama kare hata (RMSE hesabı için)

# %%
# AirPassengers veri setini pmdarima kütüphanesi yardımıyla yüklüyoruz.
# as_series=True parametresi ile veriyi bir Pandas Serisi olarak alıyoruz.
data = load_airpassengers(as_series=True)

# 1949 Ocak'tan başlayan aylık tarih indeksi ekleyelim ("MS" = ay başı).
data.index = pd.date_range(start="1949-01-01", periods=len(data), freq="MS")

# Verinin ilk beş satırını görüntüleyelim.
print(data.head())

# Veriyi görselleştirelim.
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Aylık Hava Yolu Yolcu Sayıları (1949-1960)')
plt.xlabel('Yıl')
plt.ylabel('Yolcu Sayısı')
plt.show()

# %% [markdown]
# ## 7.7.2 Durağanlık testleri ve ACF/PACF
# KPSS p-değeri tablo sınırlarının dışında kalırsa `InterpolationWarning` görülebilir; bu bir hata değildir.
# statsmodels 0.15+ ayrıca `FutureWarning` yazabilir (ileride demet yerine sonuç nesnesi dönecek); sonuçları değiştirmez.

# %%
# ADF testi (H0: birim kök var, seri durağan değil)
adf_p = adfuller(data)[1]
# KPSS testi (H0: seri durağan)
kpss_p = kpss(data, regression='c', nlags='auto')[1]
print(f"Orijinal seri  -> ADF p = {adf_p:.3f}, KPSS p = {kpss_p:.3f}")

# Log + mevsimsel fark + normal fark
data_stationary = np.log(data).diff(12).diff().dropna()
print(f"Durağanlaştırılmış seri -> ADF p = {adfuller(data_stationary)[1]:.4f}, "
      f"KPSS p = {kpss(data_stationary, regression='c', nlags='auto')[1]:.3f}")

# %%
# ACF ve PACF grafiklerini çizdirelim
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Orijinal veri için ACF grafiği
plot_acf(data, ax=ax1, lags=40)
ax1.set_title('Otokorelasyon Fonksiyonu (ACF)')

# Orijinal veri için PACF grafiği
plot_pacf(data, ax=ax2, lags=40)
ax2.set_title('Kısmi Otokorelasyon Fonksiyonu (PACF)')

plt.tight_layout()
plt.show()

# %% [markdown]
# Aynı grafikleri durağanlaştırılmış seri için çizelim: 7.6.4'teki R grafiklerinde olduğu gibi
# gecikme 1 ve 12'de belirgin negatif çubuklar görülür.

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data_stationary, ax=ax1, lags=36)
ax1.set_title('Durağanlaştırılmış Seri: ACF')
plot_pacf(data_stationary, ax=ax2, lags=36)
ax2.set_title('Durağanlaştırılmış Seri: PACF')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7.7.3 Veriyi eğitim ve test olarak ayırma
# Zaman serisinde ayrım kronolojiktir: son 60 ay (1956-1960) test seti.

# %%
# Veri setini eğitim ve test olarak ayırıyoruz. Son 60 ay test verisi olacak.
train_data = data[:-60]
test_data = data[-60:]
print(f"Eğitim: {train_data.index[0]:%Y-%m} - {train_data.index[-1]:%Y-%m} ({len(train_data)} ay)")
print(f"Test:   {test_data.index[0]:%Y-%m} - {test_data.index[-1]:%Y-%m} ({len(test_data)} ay)")

# %% [markdown]
# ## 7.7.4 `auto_arima` ile en uygun modeli bulma

# %%
# auto_arima fonksiyonunu kullanarak en uygun ARIMA modelini buluyoruz.
# seasonal=True, veride mevsimsellik olduğunu belirtir.
# m=12, mevsimsel döngünün 12 ay olduğunu (yıllık) ifade eder.
# stepwise=True, tüm kombinasyonlar yerine daha hızlı bir adımsal arama yapar.
# trace=True, denenen her modeli ve AIC değerini ekrana yazar.
auto_model = auto_arima(train_data,
                        seasonal=True,
                        m=12,
                        stepwise=True,
                        suppress_warnings=True,
                        trace=True)

# Bulunan en iyi modelin özetini yazdırıyoruz.
print(auto_model.summary())

# %% [markdown]
# ## 7.7.5 Tahmin ve değerlendirme

# %%
# Test seti için tahminler yapıyoruz. n_periods, tahmin edilecek dönem sayısını belirtir.
predictions_arima = auto_model.predict(n_periods=len(test_data))

# Tahminleri, test verisi ile aynı indekse sahip bir Pandas Serisine dönüştürelim.
predictions_arima = pd.Series(np.asarray(predictions_arima), index=test_data.index)

# Gerçek değerler ve tahminleri görselleştirelim.
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Eğitim Verisi')
plt.plot(test_data, label='Gerçek Değerler (Test)', color='orange')
plt.plot(predictions_arima, label='ARIMA Tahminleri', color='green')
plt.title('ARIMA Modeli ile Yolcu Sayısı Tahmini')
plt.xlabel('Yıl')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()

# Modelin performansını Kök Ortalama Kare Hata (RMSE) ile ölçelim.
rmse_arima = np.sqrt(mean_squared_error(test_data, predictions_arima))
print(f'ARIMA Modeli RMSE Değeri: {rmse_arima:.2f}')

# %% [markdown]
# **Alıştırma (notta 7.7.5):** Modeli `np.log(train_data)` üzerinde kurun, tahminleri `np.exp()` ile
# orijinal ölçeğe döndürün ve RMSE'yi yeniden hesaplayın. Bizim denememizde log seride
# ARIMA(2,0,0)(0,1,1)[12] seçildi ve RMSE yaklaşık 49.0 çıktı (log'suz model: 47.87).
# Hataları yıllara göre inceleyin: log model 1956-1957'de çok isabetli, 1958 sonrasında fazla tahmin yapar.
