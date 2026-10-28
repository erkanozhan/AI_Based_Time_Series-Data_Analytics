# %% [markdown]
# # Bölüm 16 — TimeSeriesSplit: Zaman Serisinde Çapraz Doğrulama
# Ders notu: Course_notes.md, Bölüm 16 (16.3 parametre demosu, 16.4 fold içinde ön işleme, 16.5 kapsamlı örnek).
# Bu dosya bölümdeki Python kodlarının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch16_timeseriessplit.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.
#
# Gerekli paketler: numpy, pandas, matplotlib, scikit-learn, tensorflow, xgboost (Colab'da hepsi kurulu).
# Not: 16.4 ve 16.5'te her biri 5 fold'luk GRU eğitimleri yapılır; CPU'da birkaç dakika sürebilir.

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
# ## 16.5 Uygulama: GRU ve XGBoost'u Aynı Fold'larda Karşılaştırmak
# İki model ve bir referans (mevsimsel naive) 16.4'teki beş fold'da (1956-1960'ın her yılı) değerlendirilir.

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
# ### 3) Ortak fold'lar
# Fold'lar ayların sıra numaraları üzerinden BİR KEZ belirlenir; üç yöntem de aynı listeyi kullanır.

# %%
# =============================================================
# 3) ORTAK FOLD'LAR (16.4 ile aynı)
# =============================================================
tscv = TimeSeriesSplit(n_splits=5, test_size=12)
folds = list(tscv.split(values))          # [(train_idx, val_idx), ...] ay numaraları
for k, (train_idx, val_idx) in enumerate(folds, 1):
    print(f"Fold {k}: eğitim {df.index[train_idx[0]]:%Y-%m} - {df.index[train_idx[-1]]:%Y-%m}, "
          f"doğrulama {df.index[val_idx[0]]:%Y-%m} - {df.index[val_idx[-1]]:%Y-%m}")

def calculate_metrics(y_true, y_pred):
    """RMSE, MAE ve MAPE (tanımlar için Bölüm 8)."""
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# %% [markdown]
# ### 4) GRU: her fold'da en iyi epoch'u bul, sonra yeniden eğit
# Erken durdurma, fold'un EĞİTİM kısmının son 12 ayıyla yapılır; fold'un doğrulama yılı hiç kullanılmaz (16.4, Not).

# %%
# =============================================================
# 4) GRU İLE ÇAPRAZ DOĞRULAMA
# =============================================================
look_back = 12     # bir tam mevsimsel döngü
inner_val = 12     # erken durdurma için fold eğitiminin son 12 ayı

def create_dataset(sequence, look_back=1):
    """Kayan pencere (Bölüm 15.1): look_back geçmiş değer girdi, sonraki değer hedef."""
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:(i + look_back), 0])
        y.append(sequence[i + look_back, 0])
    return np.array(X), np.array(y)

# GRU: 2 kapı (güncelleme z, sıfırlama r) + gizli durum; LSTM'den daha az parametre.
def build_gru_model(look_back, units=50, dropout_rate=0.2):
    model = Sequential([
        Input(shape=(look_back, 1)),               # (12 zaman adımı, 1 özellik)
        GRU(units),                                # units: gizli durumun boyutu
        Dropout(dropout_rate),                     # aşırı öğrenmeye karşı
        Dense(1)                                   # tek değerli tahmin
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

target_idx = np.arange(look_back, len(values))     # her pencerenin hedef ayı
gru_pred, best_epochs = {}, []

for k, (train_idx, val_idx) in enumerate(folds, 1):
    # a) Ölçekleyici YALNIZCA bu fold'un eğitim aylarıyla fit edilir (16.4)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[train_idx])
    X_all, y_all = create_dataset(scaler.transform(values), look_back)
    X_all = X_all.reshape(-1, look_back, 1)        # [örnek, zaman adımı, özellik]
    tr_mask = target_idx <= train_idx[-1]
    va_mask = np.isin(target_idx, val_idx)
    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va = X_all[va_mask]

    # b) Kaç epoch? Eğitimin son 12 penceresi iç doğrulama
    tf.keras.utils.set_random_seed(SEED)
    model = build_gru_model(look_back)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(X_tr[:-inner_val], y_tr[:-inner_val], epochs=200, batch_size=8,
                        validation_data=(X_tr[-inner_val:], y_tr[-inner_val:]),
                        callbacks=[early_stop], verbose=0)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    best_epochs.append(best_epoch)

    # c) Aynı mimari SIFIRDAN, fold'un bütün eğitim verisiyle best_epoch kadar eğitilir
    tf.keras.utils.set_random_seed(SEED)
    model = build_gru_model(look_back)
    model.fit(X_tr, y_tr, epochs=best_epoch, batch_size=8, verbose=0)

    # d) Doğrulama yılının tahmini, yolcu sayısı ölçeğinde
    gru_pred[k] = scaler.inverse_transform(model.predict(X_va, verbose=0)).ravel()
    print(f"Fold {k}: {len(X_tr)} eğitim penceresi, "
          f"{len(history.history['loss'])} epoch sürdü, en iyi epoch {best_epoch}")

# %% [markdown]
# ### 5) XGBoost için özellik mühendisliği (Bölüm 13)

# %%
# =============================================================
# 5) XGBOOST İÇİN ÖZELLİK MÜHENDİSLİĞİ (Bölüm 13)
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
# ### 6) XGBoost ve mevsimsel naive: aynı fold'lar
# Satırlar tarihle seçilir; böylece XGBoost'un doğrulama ayları GRU'nunkilerle birebir aynıdır.

# %%
# =============================================================
# 6) XGBOOST VE MEVSİMSEL NAIVE İLE ÇAPRAZ DOĞRULAMA
# =============================================================
# Ağaç modelleri ölçekleme gerektirmez; bu yüzden fold içinde fit edilecek
# bir ön işleme adımı yoktur. Erken durdurma da kullanmıyoruz.
xgb_pred, naive_pred = {}, {}

for k, (train_idx, val_idx) in enumerate(folds, 1):
    train_end = df.index[train_idx[-1]]                  # eğitimin son ayı
    val_dates = df.index[val_idx]                        # doğrulama yılının ayları
    X_tr, y_tr = X.loc[:train_end], y.loc[:train_end]

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
    xgb_pred[k] = model_xgb.predict(X.loc[val_dates])

    # Mevsimsel naive: "bu ay = geçen yılın aynı ayı" (model gerekmez)
    naive_pred[k] = values[val_idx - 12].ravel()
    print(f"Fold {k}: XGBoost eğitimi {len(X_tr)} gözlem "
          f"({X_tr.index[0]:%Y-%m} - {X_tr.index[-1]:%Y-%m})")

# %% [markdown]
# ### 7) Fold fold sonuçlar ve ortalama ± standart sapma

# %%
# =============================================================
# 7) SONUÇ TABLOSU
# =============================================================
models = {'GRU': gru_pred, 'XGBoost': xgb_pred, 'Mevsimsel naive': naive_pred}
rows = []
for k, (train_idx, val_idx) in enumerate(folds, 1):
    y_true = values[val_idx].ravel()
    for name, pred in models.items():
        rmse, mae, mape = calculate_metrics(y_true, pred[k])
        rows.append({'fold': k, 'yıl': df.index[val_idx[0]].year, 'model': name,
                     'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
results = pd.DataFrame(rows)

rmse_table = results.pivot(index='yıl', columns='model', values='RMSE')[list(models)]
rmse_table['GRU - XGBoost'] = rmse_table['GRU'] - rmse_table['XGBoost']
print("\nFOLD BAŞINA RMSE")
print(rmse_table.round(2).to_string())

print(f"\n{'Model':<16} {'RMSE':>15} {'MAE':>15} {'MAPE':>16}")
for name in models:
    r = results[results['model'] == name]
    print(f"{name:<16} " + " ".join(
        f"{r[m].mean():>7.2f} ± {r[m].std(ddof=0):<5.2f}" for m in ['RMSE', 'MAE']) +
        f"  {r['MAPE'].mean():>5.2f}% ± {r['MAPE'].std(ddof=0):.2f}%")

diff = rmse_table['GRU - XGBoost']
print(f"\nGRU, XGBoost'tan {int((diff < 0).sum())} fold'da daha iyi, "
      f"{int((diff > 0).sum())} fold'da daha kötü; fark ortalaması {diff.mean():.2f}")
print(f"GRU fold'larındaki en iyi epoch'lar: {best_epochs}")

# %% [markdown]
# ### 8) Grafikler: fold başına RMSE ve doğrulama yıllarının tahminleri

# %%
# =============================================================
# 8) GRAFİKLER
# =============================================================
# Doğrulama yılları ardışık olduğu için beş fold'un tahminleri
# 1956-1960 boyunca kesintisiz bir "fold dışı tahmin" çizgisi oluşturur.
val_all = np.concatenate([val_idx for _, val_idx in folds])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
rmse_table[list(models)].plot.bar(ax=axes[0], rot=0)
axes[0].set_title('Fold başına RMSE (doğrulama yılı)')
axes[0].set_ylabel('RMSE')
axes[1].plot(df.index[val_all], values[val_all], 'k-', label='Gerçek', linewidth=2)
for name, pred in models.items():
    axes[1].plot(df.index[val_all], np.concatenate([pred[k] for k in pred]), '--', label=name)
axes[1].set_title('Fold dışı tahminler (1956-1960)')
axes[1].set_ylabel('Yolcu Sayısı')
for ax in axes:
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 9) Özellik önemi (son fold'un XGBoost modeli)

# %%
# =============================================================
# 9) ÖZELLİK ÖNEMİ
# =============================================================
# model_xgb, döngünün son turunda (eğitim 1950-02 - 1959-12) eğitilen modeldir.
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Önem Skoru'); plt.title('XGBoost Özellik Önemi (Fold 5)')
plt.tight_layout()
plt.show()

print("\nEn önemli 5 özellik:")
print(feature_importance.iloc[::-1].head(5).to_string(index=False))  # en önemliden başlayarak
