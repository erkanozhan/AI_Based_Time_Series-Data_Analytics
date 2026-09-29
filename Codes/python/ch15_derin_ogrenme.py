# %% [markdown]
# # Bölüm 15 — Derin Öğrenme ile Tahmin: LSTM, GRU ve 1D-CNN
# Ders notu: Course_notes.md, Bölüm 15. Bu dosya bölümdeki Python kodlarının tamamını,
# baştan sona çalışacak sırayla içerir (15.1 ortak veri → 15.2 LSTM → 15.3 GRU → 15.4 1D-CNN → 15.5 karşılaştırma).
# Çalıştırma: depo kök dizininden `python Codes/python/ch15_derin_ogrenme.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.
#
# Gerekli paketler: numpy, pandas, matplotlib, scikit-learn, tensorflow, xgboost, pmdarima
# (Colab'da tensorflow ve xgboost kurulu; pmdarima ilk hücrede kurulur).
# Not: Eğitim CPU'da birkaç dakika sürebilir; sonuçlar tohum ve kütüphane sürümüne göre biraz değişir.

# %% [markdown]
# ## 15.1 Ortak Veri Hazırlığı
# ### 15.1.1 Veri, eğitim-test ayrımı ve ölçekleme
# Ölçekleyici yalnızca eğitim dönemiyle (ilk 84 ay) fit edilir; son 60 ay test dönemidir.

# %%
# COLAB: pip install -q pmdarima
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from pmdarima.datasets import load_airpassengers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Tekrarlanabilirlik: ağırlıkların başlangıç değerleri rastgele atanır,
# tohumları sabitleyerek her çalıştırmada benzer sonuçlar alırız.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Bölüm 7'deki Python uygulamasıyla aynı veri: 1949-1960 aylık yolcu sayıları (144 gözlem)
data = load_airpassengers(as_series=True)
dates = pd.date_range(start='1949-01-01', periods=len(data), freq='MS')  # grafikler için tarih ekseni

# Ölçekleyici 2 boyutlu dizi bekler: (gözlem sayısı, 1)
dataset = data.values.astype('float32').reshape(-1, 1)

# Bölüm 7'deki ARIMA ile aynı ayrım: son 60 ay (1956-1960) test dönemi
test_horizon = 60
train_size = len(dataset) - test_horizon   # 84 ay eğitim

# Ölçekleyici YALNIZCA eğitim dönemiyle fit edilir; test dönemine sadece dönüşüm uygulanır.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataset[:train_size])
dataset_scaled = scaler.transform(dataset)

print(f"Eğitim: {train_size} ay, Test: {test_horizon} ay")
print(f"Ölçeklenmiş eğitim aralığı: {dataset_scaled[:train_size].min():.2f} - {dataset_scaled[:train_size].max():.2f}")
print(f"Ölçeklenmiş test üst değeri: {dataset_scaled[train_size:].max():.2f}")

# %% [markdown]
# ### 15.1.2 Kayan pencere: seriden girdi-hedef çiftlerine
# `look_back = 12`; girdi tensörünün şekli [örnek, zaman adımı, özellik] (15.1.3).

# %%
def create_dataset(series, look_back=1):
    """
    Kayan pencere: her örnekte look_back geçmiş değer girdi,
    hemen sonraki değer hedef olur.

    series   : ölçeklenmiş seri, boyut (n, 1)
    Döndürür : X boyutu (n - look_back, look_back), y boyutu (n - look_back,)
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back), 0])   # girdi penceresi
        y.append(series[i + look_back, 0])       # pencerenin hemen sonraki değeri
    return np.array(X), np.array(y)

look_back = 12
X_all, y_all = create_dataset(dataset_scaled, look_back)   # (132, 12) ve (132,)

# i. örneğin hedefi serinin (i + look_back). gözlemidir.
# Hedefi test dönemine (son 60 ay) düşen örnekler test kümesine gider.
split = train_size - look_back   # 84 - 12 = 72
trainX, trainY = X_all[:split], y_all[:split]
testX, testY = X_all[split:], y_all[split:]

# Keras'ın tekrarlayan ve evrişimli katmanları 3 boyutlu girdi bekler:
# [örnek sayısı, zaman adımı sayısı, özellik sayısı]
trainX = trainX.reshape(trainX.shape[0], look_back, 1)
testX = testX.reshape(testX.shape[0], look_back, 1)
print("trainX:", trainX.shape, " testX:", testX.shape)   # (72, 12, 1)  (60, 12, 1)

# Karşılaştırmalarda kullanacağımız gerçek test değerleri (orijinal ölçek) ve tarihleri
testY_inv = scaler.inverse_transform(testY.reshape(-1, 1)).ravel()
test_dates = dates[train_size:]

# %% [markdown]
# ## 15.2 LSTM ile Tahmin

# %%
# 15.1'de hazırlanan trainX, trainY, testX, testY, scaler, dates, test_dates,
# testY_inv ve look_back değişkenlerini kullanır.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model_lstm = Sequential()
# 50: katmandaki hafıza birimi (gizli durum boyutu) sayısı.
# input_shape: (zaman adımı sayısı, özellik sayısı) = (12, 1)
model_lstm.add(LSTM(50, input_shape=(look_back, 1)))
# Tek nöronlu çıktı katmanı: bir sonraki ayın (ölçeklenmiş) değeri
model_lstm.add(Dense(1))

# Kayıp fonksiyonu: ortalama kare hata; optimizasyon: Adam
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.summary()

# epochs: eğitim verisinin model üzerinden kaç kez geçirileceği
# batch_size=1: her örnekten sonra ağırlıklar güncellenir (küçük veri için uygun, ama yavaş)
# verbose=2: her epoch için tek satır bilgi
model_lstm.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Tahminler ve orijinal ölçeğe geri dönüş
train_predict = scaler.inverse_transform(model_lstm.predict(trainX))
test_predict = scaler.inverse_transform(model_lstm.predict(testX))

rmse_lstm = np.sqrt(mean_squared_error(testY_inv, test_predict[:, 0]))
print(f'LSTM Modeli RMSE Değeri: {rmse_lstm:.2f}')

# Görselleştirme: eğitim tahminlerinin hedefleri look_back. aydan başlar
plt.figure(figsize=(15, 7))
plt.plot(dates, dataset[:, 0], label='Orijinal Veri')
plt.plot(dates[look_back:train_size], train_predict[:, 0], label='Eğitim Tahminleri (LSTM)')
plt.plot(test_dates, test_predict[:, 0], label='Test Tahminleri (LSTM)', color='orange')
plt.axvline(test_dates[0], color='gray', linestyle=':', label='Test başlangıcı')
plt.title('LSTM Modeli ile Yolcu Sayısı Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()

# %% [markdown]
# ## 15.3 GRU
# LSTM ile birebir aynı veri ve eğitim ayarları; fark yalnızca hücre yapısında.

# %%
# 15.1'de hazırlanan trainX, trainY, testX, testY, scaler, dates, test_dates,
# testY_inv ve look_back değişkenlerini kullanır.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Aynı başlangıç koşulları için tohumu yeniden sabitliyoruz
tf.random.set_seed(SEED)

model_gru = Sequential()
# 50 birimli GRU katmanı; girdi şekli LSTM'dekiyle aynı: (12 zaman adımı, 1 özellik)
model_gru.add(GRU(50, input_shape=(look_back, 1)))
model_gru.add(Dense(1))

model_gru.compile(loss='mean_squared_error', optimizer='adam')
model_gru.summary()   # GRU katmanı: 7.950 parametre (LSTM'de 10.400)

# LSTM ile aynı eğitim ayarları: adil karşılaştırma için
model_gru.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Tahminler ve orijinal ölçeğe dönüş
train_predict_gru = scaler.inverse_transform(model_gru.predict(trainX))
test_predict_gru = scaler.inverse_transform(model_gru.predict(testX))

rmse_gru = np.sqrt(mean_squared_error(testY_inv, test_predict_gru[:, 0]))
print(f'GRU Modeli RMSE Değeri: {rmse_gru:.2f}')

# Görselleştirme
plt.figure(figsize=(15, 7))
plt.plot(dates, dataset[:, 0], label='Orijinal Veri')
plt.plot(dates[look_back:train_size], train_predict_gru[:, 0], label='Eğitim Tahminleri (GRU)')
plt.plot(test_dates, test_predict_gru[:, 0], label='Test Tahminleri (GRU)', color='green')
plt.axvline(test_dates[0], color='gray', linestyle=':', label='Test başlangıcı')
plt.title('GRU Modeli ile Yolcu Sayısı Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()

# %% [markdown]
# ## 15.4 1D-CNN: Desen Tabanlı Yaklaşım
# ### 15.4.2 1D-CNN uygulaması
# Eğitim kümesinin son 12 örneği doğrulama kümesi olur; erken durdurma uygulanır.

# %%
# 15.1'de hazırlanan trainX, trainY, testX, testY, scaler, dates, test_dates,
# dataset ve look_back değişkenlerini kullanır.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Conv1D, MaxPooling1D,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

tf.random.set_seed(SEED)

# %% [markdown]
# ### 1) Eğitim / doğrulama / test ayrımı

# %%
# =============================================================
# 1) EĞİTİM / DOĞRULAMA / TEST AYIRIMI
# =============================================================
# Kronolojik sıra korunur: eğitim penceresinin son 12 örneği doğrulama olur.
#   - Eğitim   : ağırlıkları öğrenir
#   - Doğrulama: erken durdurma ve hiperparametre seçimi için
#   - Test     : son performans ölçümü (eğitimde hiç kullanılmaz)
val_size = 12
X_train, y_train = trainX[:-val_size], trainY[:-val_size]
X_val, y_val = trainX[-val_size:], trainY[-val_size:]
X_test, y_test = testX, testY

print(f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)} örnek")

# %% [markdown]
# ### 2) 1D-CNN mimarisi

# %%
# =============================================================
# 2) 1D-CNN MODELİNİN MİMARİSİ
# =============================================================
# Conv1D            : filtreler pencere üzerinde kayarak yerel desenleri öğrenir
# BatchNormalization: katman çıktısını normalize eder, eğitimi kararlı hâle getirir
# MaxPooling1D      : özellik haritasını küçültür, en belirgin sinyali korur
# Dropout           : eğitimde rastgele nöronları kapatarak aşırı öğrenmeyi azaltır
# Flatten + Dense   : öğrenilen desenleri birleştirip tek bir tahmine dönüştürür

def build_cnn_model(look_back, filters=64, kernel_size=3, dropout_rate=0.2):
    """
    Mimari:
        Conv1D → BatchNorm → MaxPool → Dropout →
        Conv1D → BatchNorm → MaxPool → Dropout →
        Flatten → Dense → Dropout → Dense (çıktı)
    """
    model = Sequential([
        # İlk evrişim bloğu; padding='same' uzunluğu korur (12)
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
               padding='same', input_shape=(look_back, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),

        # İkinci evrişim bloğu: daha fazla filtre, daha karmaşık desenler
        Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu',
               padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),

        # Düzleştirme ve tam bağlantılı katmanlar
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # regresyon çıktısı (aktivasyon yok)
    ])

    # Adam: uyarlamalı öğrenme oranı; MSE: regresyon için standart kayıp
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model_cnn = build_cnn_model(look_back, filters=64, kernel_size=3, dropout_rate=0.2)
model_cnn.summary()

# Boyut takibi (padding='same' ile):
#   Giriş     : (batch, 12, 1)
#   Conv1D_1  : (batch, 12, 64)
#   MaxPool_1 : (batch, 6, 64)     12/2 = 6
#   Conv1D_2  : (batch, 6, 128)
#   MaxPool_2 : (batch, 3, 128)    6/2 = 3
#   Flatten   : (batch, 384)       3 × 128
#   Dense_1   : (batch, 50)
#   Dense_2   : (batch, 1)

# %% [markdown]
# ### 3) Eğitim (erken durdurma) ve öğrenme eğrileri

# %%
# =============================================================
# 3) MODELİN EĞİTİLMESİ
# =============================================================
# batch_size=16: hız ile kararlılık arasında denge (batch_size=1 çok gürültülü ve yavaştır)
# EarlyStopping: doğrulama kaybı 20 epoch boyunca iyileşmezse durur ve
#                en iyi ağırlıkları geri yükler; böylece epoch sayısı otomatik belirlenir.
early_stop = EarlyStopping(monitor='val_loss', patience=20,
                           restore_best_weights=True, verbose=1)

history = model_cnn.fit(
    X_train, y_train,
    epochs=300,               # üst sınır; erken durdurma genellikle daha önce keser
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)
print(f"Eğitim {len(history.history['loss'])} epoch sürdü.")

# Öğrenme eğrileri:
#   - İki kayıp birlikte düşüyorsa: iyi
#   - Eğitim düşerken doğrulama artıyorsa: aşırı öğrenme
#   - İkisi de yüksek kalıyorsa: yetersiz öğrenme
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Eğitim Kaybı')
axes[0].plot(history.history['val_loss'], label='Doğrulama Kaybı')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE'); axes[0].set_title('Kayıp')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['mae'], label='Eğitim MAE')
axes[1].plot(history.history['val_mae'], label='Doğrulama MAE')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE'); axes[1].set_title('MAE')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4) Tahmin ve performans değerlendirmesi

# %%
# =============================================================
# 4) TAHMİN VE PERFORMANS DEĞERLENDİRMESİ
# =============================================================
# Tahminler [0, 1] ölçeğinde; inverse_transform ile orijinal ölçeğe dönüyoruz.
train_pred_inv = scaler.inverse_transform(model_cnn.predict(X_train, verbose=0))
val_pred_inv = scaler.inverse_transform(model_cnn.predict(X_val, verbose=0))
test_pred_inv = scaler.inverse_transform(model_cnn.predict(X_test, verbose=0))

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

def calculate_metrics(y_true, y_pred, set_name=""):
    """RMSE, MAE ve MAPE'yi hesaplar ve yazdırır (tanımlar için Bölüm 8)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{set_name:<15} RMSE: {rmse:7.2f}   MAE: {mae:7.2f}   MAPE: {mape:5.2f}%")
    return rmse, mae, mape

print("\n1D-CNN MODEL PERFORMANSI")
train_rmse, train_mae, train_mape = calculate_metrics(y_train_inv, train_pred_inv, "Eğitim")
val_rmse, val_mae, val_mape = calculate_metrics(y_val_inv, val_pred_inv, "Doğrulama")
test_rmse, test_mae, test_mape = calculate_metrics(y_test_inv, test_pred_inv, "Test")

rmse_cnn = test_rmse   # 15.5'teki karşılaştırmada kullanılacak

# %% [markdown]
# ### 5) Tahminlerin görselleştirilmesi

# %%
# =============================================================
# 5) TAHMİNLERİN GÖRSELLEŞTİRİLMESİ
# =============================================================
# create_dataset ilk look_back gözlemi "harcar"; tarihler buna göre kaydırılır.
train_dates = dates[look_back:look_back + len(y_train)]
val_dates = dates[look_back + len(y_train):look_back + len(y_train) + len(y_val)]

plt.figure(figsize=(14, 6))
plt.plot(dates, dataset[:, 0], 'b-', label='Gerçek Değerler', alpha=0.7)
plt.plot(train_dates, train_pred_inv, 'g--', label='Eğitim Tahminleri', alpha=0.6)
plt.plot(val_dates, val_pred_inv, color='orange', linestyle='--', label='Doğrulama Tahminleri')
plt.plot(test_dates, test_pred_inv, 'r--', label='Test Tahminleri', linewidth=2)
plt.axvline(x=val_dates[0], color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=test_dates[0], color='gray', linestyle=':', alpha=0.7)
plt.xlabel('Tarih'); plt.ylabel('Yolcu Sayısı (bin)')
plt.title('1D-CNN Model Tahminleri')
plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6) Hata analizi

# %%
# =============================================================
# 6) HATA ANALİZİ
# =============================================================
# Hataların dağılımı ve zaman içindeki seyri, modelin sistematik bir
# sapması (ör. sürekli düşük tahmin) olup olmadığını gösterir.
test_errors = y_test_inv.flatten() - test_pred_inv.flatten()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(test_errors, bins=10, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='r', linestyle='--')
axes[0].set_xlabel('Tahmin Hatası'); axes[0].set_ylabel('Frekans'); axes[0].set_title('Hata Dağılımı')

axes[1].plot(test_dates, test_errors, 'b-o')
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Tarih'); axes[1].set_ylabel('Hata'); axes[1].set_title('Hataların Zaman Seyri')
axes[1].tick_params(axis='x', rotation=45)

axes[2].scatter(y_test_inv, test_pred_inv, alpha=0.7)
lo = min(y_test_inv.min(), test_pred_inv.min())
hi = max(y_test_inv.max(), test_pred_inv.max())
axes[2].plot([lo, hi], [lo, hi], 'r--', label='Mükemmel Tahmin')   # 45 derece çizgisi
axes[2].set_xlabel('Gerçek Değerler'); axes[2].set_ylabel('Tahminler'); axes[2].set_title('Gerçek ve Tahmin')
axes[2].legend()
plt.tight_layout()
plt.show()

print(f"Ortalama hata: {np.mean(test_errors):.2f} (0'a yakın olmalı)")
print(f"Hata std     : {np.std(test_errors):.2f}")

# %% [markdown]
# ### 7) Hiperparametre karşılaştırması (isteğe bağlı)
# En iyi yapılandırma doğrulama hatasına göre seçilir. Bu hücre üç ek model eğitir; zaman kazanmak için atlayabilirsiniz.

# %%
# =============================================================
# 7) HİPERPARAMETRE KARŞILAŞTIRMASI (İSTEĞE BAĞLI)
# =============================================================
# DİKKAT: En iyi yapılandırma DOĞRULAMA hatasına göre seçilir.
# Test hatasına bakarak seçim yapmak, test kümesini dolaylı olarak
# eğitime katmak (sızıntı) anlamına gelir.
configs = [
    {'filters': 32, 'kernel_size': 2, 'dropout_rate': 0.1},
    {'filters': 64, 'kernel_size': 3, 'dropout_rate': 0.2},
    {'filters': 128, 'kernel_size': 3, 'dropout_rate': 0.3},
]

results = []
for i, config in enumerate(configs, 1):
    m = build_cnn_model(look_back, **config)
    m.fit(X_train, y_train, epochs=100, batch_size=16,
          validation_data=(X_val, y_val),
          callbacks=[EarlyStopping(patience=15, restore_best_weights=True, verbose=0)],
          verbose=0)
    val_inv = scaler.inverse_transform(m.predict(X_val, verbose=0))
    tst_inv = scaler.inverse_transform(m.predict(X_test, verbose=0))
    results.append({
        'config': str(config),
        'val_rmse': np.sqrt(mean_squared_error(y_val_inv, val_inv)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_inv, tst_inv)),
    })
    print(f"Yapılandırma {i}: {config}  ->  doğrulama RMSE: {results[-1]['val_rmse']:.2f}")

best = min(results, key=lambda r: r['val_rmse'])
print(f"\nSeçilen yapılandırma (doğrulamaya göre): {best['config']}")
print(f"Bu yapılandırmanın test RMSE değeri: {best['test_rmse']:.2f}")

# %% [markdown]
# ## 15.5 Model Karşılaştırması
# ### ARIMA (Bölüm 7.7'nin tekrarı)
# Notta `rmse_arima` Bölüm 7'deki Python uygulamasından gelir. Bu dosyanın tek başına çalışabilmesi için aynı `auto_arima` modeli burada, aynı eğitim (ilk 84 ay) ve test (son 60 ay) dönemleriyle yeniden kurulur.

# %%
# Bölüm 7.7.3-7.7.5'in kısa tekrarı: notta rmse_arima Bölüm 7'den gelir;
# bu dosya tek başına çalışsın diye aynı modeli burada yeniden kuruyoruz.
from pmdarima import auto_arima

train_data = data.iloc[:-test_horizon]   # 1949-1955 (84 ay)
test_data = data.iloc[-test_horizon:]    # 1956-1960 (60 ay)

auto_model = auto_arima(train_data, seasonal=True, m=12, stepwise=True,
                        suppress_warnings=True, trace=False)
print(auto_model)   # Bölüm 7.7.4'te seçilen model: ARIMA(1,0,0)(0,1,1)[12], sabit terimli

# 60 ay ileriye çok adımlı tahmin
predictions_arima = np.asarray(auto_model.predict(n_periods=len(test_data)))
rmse_arima = np.sqrt(mean_squared_error(test_data, predictions_arima))
print(f'ARIMA Modeli RMSE Değeri: {rmse_arima:.2f}')   # Bölüm 7.7.5: yaklaşık 47.9

# %% [markdown]
# ### XGBoost (aynı 60 aylık test dönemi) ve karşılaştırma tablosu

# %%
# Gerekli değişkenler:
#   rmse_arima                     -> yukarıdaki hücre (Bölüm 7.7'nin tekrarı, son 60 ay test)
#   rmse_lstm, rmse_gru, rmse_cnn  -> bu bölümün 15.2, 15.3 ve 15.4 kodları
#   dataset, dates, test_dates, look_back, SEED -> 15.1
import xgboost as xgb

# ---- XGBoost (Bölüm 13) aynı test dönemi için: 12 gecikme + ay bilgisi
s = pd.Series(dataset[:, 0], index=dates)
feat = pd.DataFrame({'y': s})
for lag in range(1, look_back + 1):
    feat[f'lag_{lag}'] = s.shift(lag)     # yalnızca geçmiş değerler: sızıntı yok
feat['month'] = feat.index.month
feat = feat.dropna()                      # ilk 12 ayın gecikmeleri eksik

train_f = feat[feat.index < test_dates[0]]
test_f = feat[feat.index >= test_dates[0]]   # 60 ay

model_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED)
model_xgb.fit(train_f.drop(columns='y'), train_f['y'])
pred_xgb = model_xgb.predict(test_f.drop(columns='y'))
rmse_xgb = np.sqrt(mean_squared_error(test_f['y'], pred_xgb))

# ---- Karşılaştırma tablosu
results_df = pd.DataFrame({
    'Model': ['ARIMA (Bölüm 7)', 'XGBoost (Bölüm 13)', 'LSTM', 'GRU', '1D-CNN'],
    'RMSE': [rmse_arima, rmse_xgb, rmse_lstm, rmse_gru, rmse_cnn],
}).sort_values('RMSE').reset_index(drop=True)
print(results_df.to_string(index=False, float_format='%.2f'))

# ---- Görsel karşılaştırma
plt.figure(figsize=(9, 4))
plt.barh(results_df['Model'], results_df['RMSE'], color='steelblue')
plt.gca().invert_yaxis()                  # en iyi model en üstte
plt.xlabel('Test RMSE (bin yolcu, düşük = iyi)')
plt.title('Son 60 Ay (1956-1960) için Model Karşılaştırması')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
