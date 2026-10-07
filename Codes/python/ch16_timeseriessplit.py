# %% [markdown]
# # Bölüm 16 — TimeSeriesSplit: Zaman Serisinde Çapraz Doğrulama
# Ders notu: Course_notes.md, Bölüm 16 (16.3 parametre demosu, 16.4 fold içinde ön işleme, 16.5 kapsamlı örnek).
# Bu dosya bölümdeki Python kodlarının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch16_timeseriessplit.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.
#
# Gerekli paketler: numpy, pandas, matplotlib, scikit-learn, tensorflow, xgboost (Colab'da hepsi kurulu).
# Not: 16.4'te 5 fold x 100 epoch GRU eğitimi yapılır; CPU'da birkaç dakika sürebilir.

# %%
# Veri yükleme yardımcısı (tüm bölüm dosyalarında aynı)
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/erkanozhan/AI_Based_Time_Series-Data_Analytics/main/data/"

def veri_yolu(dosya):
    """Veriyi önce yerel data/ klasöründe arar; bulamazsa GitHub'dan okur (Colab için)."""
    for aday in (Path("data") / dosya, Path("../data") / dosya, Path("../../data") / dosya):
        if aday.exists():
            return str(aday)
    return DATA_URL + dosya

# %% [markdown]
# ## 16.3 `TimeSeriesSplit` Parametreleri
# 24 gözlemlik bir dizide üç farklı ayarın fold'ları (Şekil 16.1).

# %%
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.arange(24).reshape(-1, 1)   # 24 ardışık gözlem (ör. 2 yıllık aylık veri)

def show_folds(tscv, title):
    print(title)
    for k, (tr, te) in enumerate(tscv.split(X), 1):
        print(f"  Fold {k}: eğitim {tr.min():>2}-{tr.max():>2} ({len(tr):>2} gözlem)  "
              f"doğrulama {te.min():>2}-{te.max():>2}")

show_folds(TimeSeriesSplit(n_splits=5), "Genişleyen pencere (varsayılan):")
show_folds(TimeSeriesSplit(n_splits=3, test_size=4, gap=2), "Genişleyen pencere + gap=2:")
show_folds(TimeSeriesSplit(n_splits=4, test_size=4, max_train_size=6, gap=2),
           "Kayan pencere (max_train_size=6) + gap=2:")

# %% [markdown]
# ## 16.4 Her Fold'da Ön İşleme Yalnızca Eğitim Verisiyle Fit Edilmeli
# GRU (Bölüm 15.3) ile 5 fold'lu çapraz doğrulama; her fold'da `MinMaxScaler` yalnızca o fold'un eğitim gözlemleriyle fit edilir ve model sıfırdan kurulur.

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense

SEED = 42
tf.keras.utils.set_random_seed(SEED)   # Python, NumPy ve TensorFlow tohumları

df = pd.read_csv(veri_yolu('AirPassengers.csv'), parse_dates=['Month'], index_col='Month')
if '#Passengers' in df.columns:
    df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
values = df['Passengers'].values.astype('float32').reshape(-1, 1)

look_back = 12

def create_dataset(sequence, look_back=1):
    """Kayan pencere (Bölüm 15.1): look_back geçmiş değer girdi, sonraki değer hedef."""
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:(i + look_back), 0])
        y.append(sequence[i + look_back, 0])
    return np.array(X), np.array(y)

tscv = TimeSeriesSplit(n_splits=5, test_size=12)
fold_rmse = []

for k, (train_idx, val_idx) in enumerate(tscv.split(values), 1):
    # 1) Ölçekleyici YALNIZCA bu fold'un eğitim gözlemleriyle fit edilir
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[train_idx])
    scaled = scaler.transform(values)   # dönüşüm tüm seriye uygulanabilir; öğrenilen min/max yalnızca eğitimden

    # 2) Pencereler: i. örneğin hedefi serinin (i + look_back). gözlemidir
    X_all, y_all = create_dataset(scaled, look_back)
    target_idx = np.arange(look_back, len(values))
    tr_mask = target_idx <= train_idx[-1]      # hedefi eğitim döneminde olanlar
    va_mask = np.isin(target_idx, val_idx)     # hedefi doğrulama döneminde olanlar
    X_tr = X_all[tr_mask].reshape(-1, look_back, 1)
    X_va = X_all[va_mask].reshape(-1, look_back, 1)
    y_tr, y_va = y_all[tr_mask], y_all[va_mask]

    # 3) Her fold'da SIFIRDAN yeni bir model (önceki fold'un ağırlıkları taşınmaz)
    model = Sequential([Input(shape=(look_back, 1)), GRU(50), Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_tr, y_tr, epochs=100, batch_size=8, verbose=0)

    # 4) Tahmini orijinal ölçeğe döndürüp hatayı hesapla
    pred = scaler.inverse_transform(model.predict(X_va, verbose=0))
    true = scaler.inverse_transform(y_va.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(true, pred))
    fold_rmse.append(rmse)
    print(f"Fold {k}: eğitim {df.index[train_idx[0]]:%Y-%m} - {df.index[train_idx[-1]]:%Y-%m}, "
          f"doğrulama {df.index[val_idx[0]]:%Y-%m} - {df.index[val_idx[-1]]:%Y-%m}, RMSE = {rmse:.2f}")

print(f"\nGRU çapraz doğrulama RMSE: {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")

# %% [markdown]
# ## 16.5 Uygulama: GRU ve XGBoost ile Kapsamlı Bir Örnek

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import xgboost as xgb

# %% [markdown]
# ### 0) Tekrarlanabilirlik

# %%
# =============================================================
# 0) TEKRARLANABİLİRLİK
# =============================================================
# Ağırlıkların başlangıç değerleri ve alt örneklemler rastgeledir;
# tohumları sabitleyerek sonuçları karşılaştırılabilir kılıyoruz.
SEED = 42
tf.keras.utils.set_random_seed(SEED)   # Python, NumPy ve TensorFlow tohumları
# GPU kullanılıyorsa tam deterministik işlemler için:
# tf.config.experimental.enable_op_determinism()

# %% [markdown]
# ### 1) Veri setini yükleme

# %%
# =============================================================
# 1) VERİ SETİNİ YÜKLEME
# =============================================================
# AirPassengers: 1949-1960 aylık uluslararası havayolu yolcu sayıları (144 gözlem).
# Hem trend hem güçlü mevsimsellik içerdiği için klasik bir karşılaştırma verisidir.
df = pd.read_csv(veri_yolu('AirPassengers.csv'))
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
if '#Passengers' in df.columns:           # bazı sürümlerde sütun adı farklıdır
    df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)

values = df['Passengers'].values.astype('float32').reshape(-1, 1)
print(f"Toplam gözlem sayısı: {len(df)}")
print(df.describe())

# %% [markdown]
# ### 2) Veriyi görselleştirme

# %%
# =============================================================
# 2) VERİYİ GÖRSELLEŞTİRME
# =============================================================
# Yukarı yönlü trend, 12 aylık mevsimsellik ve zamanla artan varyans
# (Bölüm 2.4: çarpımsal yapı) grafikte açıkça görülür.
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Passengers'], linewidth=1)
plt.title('Aylık Havayolu Yolcu Sayısı (1949-1960)')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı (bin)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3) GRU için veri hazırlama
# Ölçekleyici yalnızca eğitim dönemiyle fit edilir; son 24 ay test, eğitimin son 12 ayı doğrulama.

# %%
# =============================================================
# 3) GRU İÇİN VERİ HAZIRLAMA
# =============================================================
test_size = 24     # son 2 yıl (24 ay) test için
look_back = 12     # bir tam mevsimsel döngü

# Ölçekleyici YALNIZCA eğitim dönemiyle fit edilir (16.4); test dönemine sadece
# dönüşüm uygulanır. Tahminler daha sonra inverse_transform ile geri çevrilir.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(values[:-test_size])
values_scaled = scaler.transform(values)

def create_dataset(sequence, look_back=1):
    """
    Kayan pencere (Bölüm 15.1):
        X: (n_örnek, look_back) girdi matrisi
        y: (n_örnek,) hedef vektörü; i. örneğin hedefi (i + look_back). gözlem
    """
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:(i + look_back), 0])
        y.append(sequence[i + look_back, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_dataset(values_scaled, look_back)   # (132, 12), (132,)

# Keras tekrarlayan katmanları [örnek, zaman adımı, özellik] şeklinde girdi bekler
X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

# Kronolojik eğitim / test ayrımı: son 24 örneğin hedefleri son 24 aydır
train_size = X_all.shape[0] - test_size
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# Eğitimin son 12 ayı erken durdurma için doğrulama kümesi
val_size = 12
X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]

print(f"Eğitim (final): {len(X_train_final)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")

# %% [markdown]
# ### 4) GRU modelinin kurulması

# %%
# =============================================================
# 4) GRU MODELİNİN KURULMASI
# =============================================================
# GRU: 2 kapı (güncelleme z, sıfırlama r) + gizli durum; LSTM'den daha az parametre.
def build_gru_model(look_back, units=50, dropout_rate=0.2):
    model = Sequential([
        Input(shape=(look_back, 1)),               # (12 zaman adımı, 1 özellik)
        GRU(units),                                # units: gizli durumun boyutu
        Dropout(dropout_rate),                     # aşırı öğrenmeye karşı
        Dense(1)                                   # tek değerli tahmin
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

model_gru = build_gru_model(look_back, units=50, dropout_rate=0.2)
model_gru.summary()
# Parametre sayısı (Keras varsayılanı reset_after=True, iki yanlılık vektörü):
#   GRU  : 3 * [units * (input_dim + units) + 2 * units] = 3 * [50 * 51 + 100] = 7950
#   Dense: 50 * 1 + 1 = 51
#   Toplam: 8001

# %% [markdown]
# ### 5) GRU modelinin eğitilmesi (erken durdurma)

# %%
# =============================================================
# 5) GRU MODELİNİN EĞİTİLMESİ
# =============================================================
# batch_size=8: küçük veri için hız-kararlılık dengesi
# EarlyStopping: doğrulama kaybı 15 epoch iyileşmezse durur, en iyi ağırlıkları geri yükler
early_stop = EarlyStopping(monitor='val_loss', patience=15,
                           restore_best_weights=True, verbose=1)

history = model_gru.fit(
    X_train_final, y_train_final,
    epochs=200,
    batch_size=8,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Öğrenme eğrileri: iki kayıp birlikte düşüyorsa iyi; eğitim düşerken
# doğrulama artıyorsa aşırı öğrenme vardır.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Eğitim Süreci - Kayıp')
plt.legend(); plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Eğitim MAE')
plt.plot(history.history['val_mae'], label='Doğrulama MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.title('Eğitim Süreci - MAE')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Doğrulama kaybının en düşük olduğu epoch (erken durdurma bundan 'patience' kadar sonra durur)
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
print(f"\nEğitim {len(history.history['loss'])} epoch sürdü; en iyi epoch: {best_epoch}")

# %% [markdown]
# ### 6) GRU ile tahmin ve performans değerlendirmesi
# Son model, doğrulama dahil tüm eğitim verisiyle en iyi epoch sayısı kadar eğitilir.

# %%
# =============================================================
# 6) GRU İLE TAHMİN VE PERFORMANS DEĞERLENDİRMESİ
# =============================================================
# Son model: doğrulama dahil tüm eğitim verisiyle, en iyi epoch sayısı kadar eğitilir.
tf.keras.utils.set_random_seed(SEED)
model_gru_final = build_gru_model(look_back, units=50, dropout_rate=0.2)
model_gru_final.fit(X_train, y_train, epochs=best_epoch, batch_size=8, verbose=0)

train_pred_inv = scaler.inverse_transform(model_gru_final.predict(X_train, verbose=0))
test_pred_inv = scaler.inverse_transform(model_gru_final.predict(X_test, verbose=0))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

def calculate_metrics(y_true, y_pred, set_name=""):
    """RMSE, MAE ve MAPE (tanımlar için Bölüm 8)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{set_name:<15} RMSE: {rmse:7.2f}   MAE: {mae:7.2f}   MAPE: {mape:5.2f}%")
    return rmse, mae, mape

rmse_train, mae_train, mape_train = calculate_metrics(
    y_train_inv.flatten(), train_pred_inv.flatten(), "GRU Eğitim")
rmse_test, mae_test, mape_test = calculate_metrics(
    y_test_inv.flatten(), test_pred_inv.flatten(), "GRU Test")

# create_dataset ilk look_back gözlemi "harcar"; tarihler buna göre kaydırılır
train_dates = df.index[look_back:look_back + len(y_train_inv)]
test_dates = df.index[look_back + len(y_train_inv):]

plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Passengers'], 'b-', label='Gerçek Değerler', alpha=0.7)
plt.plot(train_dates, train_pred_inv, 'g--', label='Eğitim Tahminleri', alpha=0.7)
plt.plot(test_dates, test_pred_inv, 'r--', label='Test Tahminleri', linewidth=2)
plt.axvline(x=test_dates[0], color='gray', linestyle=':', label='Test Başlangıcı')
plt.xlabel('Tarih'); plt.ylabel('Yolcu Sayısı (bin)'); plt.title('GRU Model Tahminleri')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7) XGBoost için özellik mühendisliği (Bölüm 13)

# %%
# =============================================================
# 7) XGBOOST İÇİN ÖZELLİK MÜHENDİSLİĞİ (Bölüm 13)
# =============================================================
# Ağaç modelleri seriyi doğrudan işleyemez; geçmişi özelliklere çeviririz.
# KURAL: t anındaki satırın özellikleri yalnızca t-1 ve öncesinden gelmelidir.
df_features = df.copy()

# Gecikme özellikleri: 1-12 ay önceki değerler
for lag in range(1, 13):
    df_features[f'lag_{lag}'] = df_features['Passengers'].shift(lag)

# Hareketli istatistikler: shift(1) sayesinde yalnızca geçmiş aylar kullanılır
past = df_features['Passengers'].shift(1)
df_features['rolling_mean_3'] = past.rolling(window=3).mean()
df_features['rolling_mean_6'] = past.rolling(window=6).mean()
df_features['rolling_mean_12'] = past.rolling(window=12).mean()
df_features['rolling_std_3'] = past.rolling(window=3).std()
df_features['rolling_std_12'] = past.rolling(window=12).std()

# Mevsimsel fark: geçen ayın, bir yıl önceki aynı aya göre değişimi.
# (Passengers - Passengers.shift(12)) yazmak hedefin kendisini içerdiği için sızıntı olurdu.
df_features['seasonal_diff'] = past - past.shift(12)

# Takvim özellikleri
df_features['month'] = df_features.index.month
df_features['quarter'] = df_features.index.quarter
df_features['year'] = df_features.index.year
df_features['year_normalized'] = (df_features['year'] - df_features['year'].min()) / \
                                  (df_features['year'].max() - df_features['year'].min())

# Gecikmeler nedeniyle ilk satırlarda oluşan eksik değerleri at
df_features = df_features.dropna()
feature_cols = [col for col in df_features.columns if col != 'Passengers']
print(f"\nÖzellik mühendisliği sonrası gözlem sayısı: {len(df_features)}")
print(f"Özellik sayısı: {len(feature_cols)}")

X = df_features[feature_cols]
y = df_features['Passengers']

# %% [markdown]
# ### 8) TimeSeriesSplit ile çapraz doğrulama (XGBoost, 5 fold)

# %%
# =============================================================
# 8) TIMESERIESSPLIT İLE ÇAPRAZ DOĞRULAMA
# =============================================================
# Genişleyen pencere (16.2): her fold'da eğitim büyür, doğrulama hep gelecekte kalır.
#   Fold 1: Eğitim [----]       Doğrulama [--]
#   Fold 2: Eğitim [------]     Doğrulama [--]
#   Fold 3: Eğitim [--------]   Doğrulama [--]
# Ağaç modelleri ölçekleme gerektirmez; bu yüzden fold içinde fit edilecek
# bir ön işleme adımı yoktur. Erken durdurma da kullanmıyoruz: doğrulama fold'u
# üzerinde durdurmak o fold'un hatasını iyimser gösterirdi (16.4).
tscv = TimeSeriesSplit(n_splits=5)

rmse_list, mae_list, mape_list = [], [], []
print("\nXGBOOST - TIMESERIESSPLIT ÇAPRAZ DOĞRULAMA")

for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
    X_tr, X_va = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_va = y.iloc[train_index], y.iloc[val_index]

    print(f"Fold {fold}: eğitim {len(X_tr)} gözlem ({y_tr.index.min():%Y-%m} - {y_tr.index.max():%Y-%m}), "
          f"doğrulama {len(X_va)} gözlem ({y_va.index.min():%Y-%m} - {y_va.index.max():%Y-%m})")

    # Her fold'da sıfırdan yeni model
    model_xgb = xgb.XGBRegressor(
        n_estimators=500,        # ağaç sayısı
        learning_rate=0.05,      # her ağacın katkı oranı
        max_depth=4,             # ağaç derinliği
        subsample=0.8,           # her ağaçta kullanılan satır oranı
        colsample_bytree=0.8,    # her ağaçta kullanılan özellik oranı
        random_state=SEED
    )
    model_xgb.fit(X_tr, y_tr)
    y_va_pred = model_xgb.predict(X_va)

    rmse = np.sqrt(mean_squared_error(y_va, y_va_pred))
    mae = mean_absolute_error(y_va, y_va_pred)
    mape = np.mean(np.abs((y_va - y_va_pred) / y_va)) * 100
    rmse_list.append(rmse); mae_list.append(mae); mape_list.append(mape)
    print(f"         RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

print("\nÇAPRAZ DOĞRULAMA SONUÇLARI (Ortalama ± Std)")
print(f"RMSE: {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f}")
print(f"MAE:  {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
print(f"MAPE: {np.mean(mape_list):.2f}% ± {np.std(mape_list):.2f}%")

# %% [markdown]
# ### 9) Son XGBoost modeli ve özellik önemi

# %%
# =============================================================
# 9) SON XGBOOST MODELİ VE ÖZELLİK ÖNEMİ
# =============================================================
# Çapraz doğrulama performans tahminini verdi. Şimdi GRU ile aynı son 24 ayı
# test için ayırıp son modeli eğitiyoruz.
train_end = len(X) - 24
X_train_xgb, X_test_xgb = X.iloc[:train_end], X.iloc[train_end:]
y_train_xgb, y_test_xgb = y.iloc[:train_end], y.iloc[train_end:]

final_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED)
final_xgb.fit(X_train_xgb, y_train_xgb)
y_test_pred_xgb = final_xgb.predict(X_test_xgb)

print("\nXGBOOST TEST PERFORMANSI")
xgb_rmse, xgb_mae, xgb_mape = calculate_metrics(y_test_xgb.values, y_test_pred_xgb, "XGBoost Test")

# Özellik önemi: hangi özellikler tahmine en çok katkı sağlıyor?
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_xgb.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Önem Skoru'); plt.title('XGBoost Özellik Önemi')
plt.tight_layout()
plt.show()

print("\nEn önemli 5 özellik:")
print(feature_importance.tail(5).to_string(index=False))

# %% [markdown]
# ### 10) Model karşılaştırması (aynı son 24 ay)

# %%
# =============================================================
# 10) MODEL KARŞILAŞTIRMASI (aynı son 24 ay)
# =============================================================
print(f"\n{'Model':<15} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
print("-" * 48)
print(f"{'GRU':<15} {rmse_test:>10.2f} {mae_test:>10.2f} {mape_test:>9.2f}%")
print(f"{'XGBoost':<15} {xgb_rmse:>10.2f} {xgb_mae:>10.2f} {xgb_mape:>9.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(test_dates, y_test_inv, 'b-', label='Gerçek', linewidth=2)
axes[0].plot(test_dates, test_pred_inv, 'r--', label='GRU Tahmini', linewidth=2)
axes[0].set_title(f'GRU Tahminleri (RMSE: {rmse_test:.2f})')
axes[1].plot(y_test_xgb.index, y_test_xgb.values, 'b-', label='Gerçek', linewidth=2)
axes[1].plot(y_test_xgb.index, y_test_pred_xgb, 'r--', label='XGBoost Tahmini', linewidth=2)
axes[1].set_title(f'XGBoost Tahminleri (RMSE: {xgb_rmse:.2f})')
for ax in axes:
    ax.set_xlabel('Tarih'); ax.set_ylabel('Yolcu Sayısı')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
