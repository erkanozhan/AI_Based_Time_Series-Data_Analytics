# %% [markdown]
# # Bölüm 7 — Python ile SARIMA: `AirPassengers`
# Ders notu: notlar/07_klasik_istatistiksel_modeller_arima_ve_sarima.md, Bölüm 7.7 (Python ile Aynı Analiz) ve 7.8.6 (ETS, Holt-Winters ve Theta). Bu dosya bölümdeki Python kodlarının tamamını içerir.
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

# %% [markdown]
# ## 7.8.6 Üstel düzeltme (ETS), Holt-Winters ve Theta
# Aynı ayrım (1949-1955 eğitim, 1956-1960 test) üzerinde üç yöntemi 7.7.5'teki ARIMA
# ve mevsimsel naive ile karşılaştırıyoruz.

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Holt-Winters
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # ETS (durum uzayı)
from statsmodels.tsa.forecasting.theta import ThetaModel  # Theta yöntemi

h = len(test_data)

# Holt-Winters: toplamsal trend, çarpımsal mevsimsellik
hw_model = ExponentialSmoothing(train_data, trend="add", seasonal="mul",
                                seasonal_periods=12).fit()
pred_hw = hw_model.forecast(h)

# ETS(M,Ad,M): çarpımsal hata, sönümlü trend, çarpımsal mevsimsellik
ets_model = ETSModel(train_data, error="mul", trend="add", damped_trend=True,
                     seasonal="mul", seasonal_periods=12).fit(disp=False)
pred_ets = ets_model.forecast(h)

# Theta: mevsimsellikten arındır, SES + sürüklenme uygula, mevsimselliği geri ekle
theta_model = ThetaModel(train_data, period=12).fit()
pred_theta = theta_model.forecast(h)
print(theta_model.params.round(3))  # b0: doğrunun eğimi, alpha: SES katsayısı

# Mevsimsel naive: 1955'in 12 ayını 5 kez tekrarla
pred_snaive = np.tile(train_data[-12:].values, h // 12)


def metrikler(gercek, tahmin):
    """MAE, RMSE ve MAPE (Bölüm 8.2) değerlerini sözlük olarak döndürür."""
    gercek, tahmin = np.asarray(gercek), np.asarray(tahmin)
    e = gercek - tahmin
    return {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "MAPE": np.mean(np.abs(e / gercek)) * 100}


tablo = pd.DataFrame({"Mevsimsel naive": metrikler(test_data, pred_snaive),
                      "ARIMA (7.7.5)": metrikler(test_data, predictions_arima),
                      "ETS(M,Ad,M)": metrikler(test_data, pred_ets),
                      "Holt-Winters çarpımsal": metrikler(test_data, pred_hw),
                      "Theta": metrikler(test_data, pred_theta)}).T
print(tablo.round(2))

# %%
# Yıllara göre ortalama hata (gerçek - tahmin); artı değer = model geride kalmış
hatalar = pd.DataFrame({"ARIMA": test_data - predictions_arima,
                        "ETS": test_data - pred_ets,
                        "Holt-Winters": test_data - pred_hw,
                        "Theta": test_data - pred_theta})
print(hatalar.groupby(hatalar.index.year).mean().round(1))

# Tahminleri test dönemiyle birlikte çizelim
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Eğitim Verisi', color='black')
plt.plot(test_data, label='Gerçek Değerler (Test)', color='orange')
plt.plot(predictions_arima, label='ARIMA', color='green')
plt.plot(pred_ets, label='ETS(M,Ad,M)', linestyle='--')
plt.plot(pred_hw, label='Holt-Winters çarpımsal')
plt.plot(pred_theta, label='Theta', linestyle=':')
plt.title('ARIMA, ETS, Holt-Winters ve Theta Tahminleri (Test: 1956-1960)')
plt.xlabel('Yıl')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()
