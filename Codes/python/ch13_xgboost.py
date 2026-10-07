# %% [markdown]
# # Bölüm 13 — XGBoost ile Zaman Serisi Tahmini
# Ders notu: Course_notes.md, Bölüm 13 (13.4 Python ile XGBoost Uygulaması ve 13.4.2 Fark Üzerinden Tahmin).
# Bu dosya bölümdeki Python kodlarının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch13_xgboost.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.
#
# Gerekli paketler: numpy, pandas, matplotlib, scikit-learn, xgboost (Colab'da hepsi kurulu).

# %%
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/erkanozhan/AI_Based_Time_Series-Data_Analytics/main/data/"

def veri_yolu(dosya):
    """Veriyi önce yerel data/ klasöründe arar; bulamazsa GitHub'dan okur (Colab için)."""
    for aday in (Path("data") / dosya, Path("../data") / dosya, Path("../../data") / dosya):
        if aday.exists():
            return str(aday)
    return DATA_URL + dosya

# %% [markdown]
# ## 13.4 Python ile XGBoost Uygulaması
# ### Tekrarlanabilirlik
# Rastgele örnekleme içeren adımlar için tohum sabitlenir.

# %%
# =============================================================
# 0) TEKRARLANABİLİRLİK
# =============================================================
#
# XGBoost içinde rastgele işlemler vardır (satır ve sütun örnekleme).
# Aynı sonuçları elde etmek için seed ayarlamak gerekir.

SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ### 1) Veri yükleme ve inceleme
# AirPassengers: 1949–1960 aylık yolcu sayıları (144 gözlem).

# %%
# =============================================================
# 1) VERİ YÜKLEME VE İNCELEME
# =============================================================

df = pd.read_csv(veri_yolu('AirPassengers.csv'))

# Sütun adını düzeltelim (bazı sürümlerde '#Passengers' olarak gelir)
if '#Passengers' in df.columns:
    df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)

# Tarih indeksini ayarlayalım
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

print("Veri seti özeti:")
print(f"  Gözlem sayısı: {len(df)}")
print(f"  Tarih aralığı: {df.index.min()} - {df.index.max()}")
print(f"\n{df.head()}")

# %% [markdown]
# ### 2) Özellik mühendisliği (13.2)
# Gecikmeler, hareketli istatistikler, yıllık değişim ve takvim özellikleri. Her özellik yalnızca geçmiş bilgiden üretilir.

# %%
# =============================================================
# 2) ÖZELLİK MÜHENDİSLİĞİ
# =============================================================
#
# XGBoost zaman serisini doğrudan işleyemez. Veriyi şu formata
# dönüştürmemiz gerekir (bkz. Bölüm 12):
#
#   Özellikler (X)              →  Hedef (y)
#   [lag_1, lag_2, ..., ay]     →  Passengers
#
# ALTIN KURAL: t anındaki bir özellik, yalnızca t anından ÖNCE
# bilinen bilgilerle hesaplanmalıdır. Aksi hâlde model cevabı
# "kopya çeker" (veri sızıntısı, data leakage).

df_features = df.copy()

# ---------------------------------------------------------
# Gecikme (Lag) Özellikleri
# ---------------------------------------------------------
# Geçmiş değerler en önemli özelliklerdir. Mevsimsel veri için
# en az bir tam döngü (12 ay) geriye bakmak faydalıdır.

for lag in range(1, 13):
    df_features[f'lag_{lag}'] = df_features['Passengers'].shift(lag)

# ---------------------------------------------------------
# Hareketli İstatistikler
# ---------------------------------------------------------
# Hareketli ortalama trendi, hareketli standart sapma
# volatiliteyi (dalgalanmayı) yakalar.
#
# shift(1) ile bir dönem kaydırıyoruz çünkü tahmin anında
# o anın değerini bilemeyiz.

df_features['rolling_mean_3'] = df_features['Passengers'].shift(1).rolling(3).mean()
df_features['rolling_mean_6'] = df_features['Passengers'].shift(1).rolling(6).mean()
df_features['rolling_mean_12'] = df_features['Passengers'].shift(1).rolling(12).mean()

df_features['rolling_std_3'] = df_features['Passengers'].shift(1).rolling(3).std()
df_features['rolling_std_12'] = df_features['Passengers'].shift(1).rolling(12).std()

# ---------------------------------------------------------
# Yıllık Değişim (gecikmeli mevsimsel fark)
# ---------------------------------------------------------
# Son bilinen ayın (t-1), bir yıl önceki aynı aya (t-13) göre
# değişimi. Yıllık büyüme hızını yakalar.
# DİKKAT: Passengers(t) - Passengers(t-12) yazsaydık hedef değeri
# özelliğin içine gizlemiş olurduk (veri sızıntısı).

df_features['seasonal_diff'] = (
    df_features['Passengers'].shift(1) - df_features['Passengers'].shift(13)
)

# ---------------------------------------------------------
# Takvim Özellikleri
# ---------------------------------------------------------
# Ay ve çeyrek bilgisi mevsimselliği yakalamaya yardımcı olur.

df_features['month'] = df_features.index.month
df_features['quarter'] = df_features.index.quarter

# Yıl bilgisini normalize edelim (trend için).
# Not: Ağaçlar bu özelliğin eğitimde görmediği değerlerine
# ekstrapolasyon yapamaz (bkz. 13.3).
df_features['year_normalized'] = (
    (df_features.index.year - df_features.index.year.min()) /
    (df_features.index.year.max() - df_features.index.year.min())
)

# ---------------------------------------------------------
# Eksik Değerleri Temizleme
# ---------------------------------------------------------
# Gecikme ve hareketli ortalamalar nedeniyle ilk satırlarda
# NaN oluşur. Bunları çıkarıyoruz.

df_features = df_features.dropna()

print(f"\nÖzellik mühendisliği sonrası:")
print(f"  Gözlem sayısı: {len(df_features)}")
print(f"  Özellik sayısı: {len(df_features.columns) - 1}")

# Özellik ve hedef değişkenleri ayıralım
feature_cols = [col for col in df_features.columns if col != 'Passengers']
X = df_features[feature_cols]
y = df_features['Passengers']

print(f"\nKullanılan özellikler:\n  {feature_cols}")

# %% [markdown]
# ### 3) Eğitim / doğrulama / test ayrımı
# Kronolojik: son 12 ay test, eğitimin son 12 ayı erken durdurma için doğrulama.

# %%
# =============================================================
# 3) EĞİTİM / DOĞRULAMA / TEST AYIRIMI
# =============================================================
#
# Zaman serilerinde kronolojik sıra korunmalıdır (bkz. Bölüm 8).
# Son 12 ay: test. Eğitimin son 12 ayı: erken durdurma için
# doğrulama. Test seti model seçiminde KULLANILMAZ.

test_size = 12
val_size = 12
split_point = len(X) - test_size

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

X_tr_in, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
y_tr_in, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

print(f"\nVeri bölümü:")
print(f"  Eğitim: {len(X_train)} gözlem ({y_train.index.min()} - {y_train.index.max()})")
print(f"    (bunun son {val_size} ayı erken durdurma için doğrulama)")
print(f"  Test: {len(X_test)} gözlem ({y_test.index.min()} - {y_test.index.max()})")

# %% [markdown]
# ### 4) Modelin kurulması: erken durdurma ve yeniden eğitim

# %%
# =============================================================
# 4) XGBOOST MODELİNİN KURULMASI VE EĞİTİLMESİ
# =============================================================
#
# Hiperparametrelerin anlamı için 13.1.4'teki tabloya bakınız.
#
# Adım 1: Doğrulama seti üzerinde erken durdurma ile uygun
#         ağaç sayısını bul.
# Adım 2: Bu ağaç sayısıyla modeli tüm eğitim verisinde yeniden eğit.

params = dict(
    learning_rate=0.05,       # Öğrenme hızı (eta)
    max_depth=4,              # Ağaç derinliği
    subsample=0.8,            # Veri örnekleme oranı
    colsample_bytree=0.8,     # Özellik örnekleme oranı
    random_state=SEED,
)

print("\nXGBoost modeli eğitiliyor (erken durdurma)...")
# Not: xgboost >= 2.0'da early_stopping_rounds kurucuya (XGBRegressor) yazılır;
# eski kodlardaki fit(..., early_stopping_rounds=50) biçimi TypeError verir.
es_model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, **params)
es_model.fit(X_tr_in, y_tr_in, eval_set=[(X_val, y_val)], verbose=False)

best_n = es_model.best_iteration + 1   # best_iteration 0'dan sayar: ağaç sayısı = +1
print(f"Seçilen ağaç sayısı: {best_n}")

model = xgb.XGBRegressor(n_estimators=best_n, **params)
model.fit(X_train, y_train)

# %% [markdown]
# ### 5) Tahmin ve performans değerlendirmesi (13.4.1)
# Ekstrapolasyon kontrolü: testteki en büyük tahmin (544) eğitimdeki tavanın (559) bile altında kalır.

# %%
# =============================================================
# 5) TAHMİN VE PERFORMANS DEĞERLENDİRMESİ
# =============================================================

# Tahminler
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ---------------------------------------------------------
# Performans metrikleri (tanımlar için bkz. Bölüm 8)
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred, set_name=""):
    """
    Tahmin performans metriklerini hesaplar.

    MAE: Ortalama mutlak hata - tüm hatalara eşit ağırlık
    RMSE: Kök ortalama kare hata - büyük hataları cezalandırır
    MAPE: Ortalama mutlak yüzde hata - ölçekten bağımsız
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{set_name} Performansı:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")

    return mae, rmse, mape

print("\n" + "=" * 50)
print("XGBOOST MODEL PERFORMANSI")
print("=" * 50)

train_mae, train_rmse, train_mape = calculate_metrics(
    y_train, y_train_pred, "Eğitim Seti"
)
test_mae, test_rmse, test_mape = calculate_metrics(
    y_test, y_test_pred, "Test Seti"
)

# Ekstrapolasyon kontrolü: test dönemindeki gerçek değerler
# eğitimde görülen en büyük değeri aşıyor mu?
print(f"\nEğitimdeki en büyük değer : {y_train.max():.0f}")
print(f"Testteki en büyük değer   : {y_test.max():.0f}")
print(f"Testteki en büyük tahmin  : {y_test_pred.max():.0f}")

# %% [markdown]
# ### 6) Özellik önemi

# %%
# =============================================================
# 6) ÖZELLİK ÖNEMİ ANALİZİ
# =============================================================
#
# XGBoost'un avantajlarından biri yorumlanabilirliğidir.
# Hangi özelliklerin tahmine en çok katkı sağladığını görebiliriz.
#
# Bu bilgi şu sorulara yanıt verir:
#   - Hangi gecikmeler daha önemli?
#   - Mevsimsellik mi trend mi baskın?
#   - Gereksiz özellikler var mı?

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Önem Skoru')
plt.title('XGBoost Özellik Önemi')
plt.tight_layout()
plt.show()

print("\nEn önemli 5 özellik:")
print(feature_importance.tail(5).to_string(index=False))

# %% [markdown]
# ### 7) Tahminlerin görselleştirilmesi

# %%
# =============================================================
# 7) TAHMİNLERİN GÖRSELLEŞTİRİLMESİ
# =============================================================

plt.figure(figsize=(12, 5))

# Tüm gerçek değerler
plt.plot(df.index, df['Passengers'], 'b-', label='Gerçek Değerler', alpha=0.7)

# Eğitim tahminleri
plt.plot(y_train.index, y_train_pred, 'g--', label='Eğitim Tahminleri', alpha=0.5)

# Test tahminleri
plt.plot(y_test.index, y_test_pred, 'r--', label='Test Tahminleri', linewidth=2)

# Test dönemini işaretle
plt.axvline(x=y_test.index[0], color='gray', linestyle=':', alpha=0.7)

plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı (bin)')
plt.title(f'XGBoost Tahminleri (Test RMSE: {test_rmse:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Test dönemi detaylı görünüm
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, 'b-o', label='Gerçek Değerler', linewidth=2)
plt.plot(y_test.index, y_test_pred, 'r--s', label='XGBoost Tahminleri', linewidth=2)
plt.axhline(y=y_train.max(), color='gray', linestyle=':', label='Eğitimdeki en büyük değer')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı (bin)')
plt.title('Test Dönemi Detaylı Görünüm')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8) Zaman serisi çapraz doğrulaması (TimeSeriesSplit, ayrıntısı Bölüm 16)

# %%
# =============================================================
# 8) ÇAPRAZ DOĞRULAMA (TIMESERIESSPLIT)
# =============================================================
#
# Tek bir eğitim/test bölümü yanıltıcı olabilir; test dönemi
# şanslı veya şanssız bir dönem olabilir. TimeSeriesSplit ile
# birden fazla kronolojik bölüm oluşturup modelin tutarlılığını
# ölçeriz (ayrıntısı Bölüm 16'da).
#
# Katlarda erken durdurma yapmıyoruz; doğrulama katını model
# seçiminde kullanmak skorları iyimser gösterir. Bunun yerine
# yukarıda bulunan ağaç sayısını (best_n) sabit tutuyoruz.

print("\n" + "=" * 50)
print("ÇAPRAZ DOĞRULAMA (TimeSeriesSplit)")
print("=" * 50)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = {'rmse': [], 'mae': [], 'mape': []}

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

    cv_model = xgb.XGBRegressor(n_estimators=best_n, **params)
    cv_model.fit(X_tr, y_tr)
    y_va_pred = cv_model.predict(X_va)

    # Metrikler
    rmse = np.sqrt(mean_squared_error(y_va, y_va_pred))
    mae = mean_absolute_error(y_va, y_va_pred)
    mape = np.mean(np.abs((y_va - y_va_pred) / y_va)) * 100

    cv_scores['rmse'].append(rmse)
    cv_scores['mae'].append(mae)
    cv_scores['mape'].append(mape)

    print(f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

print(f"\nOrtalama Sonuçlar:")
print(f"  RMSE: {np.mean(cv_scores['rmse']):.2f} ± {np.std(cv_scores['rmse']):.2f}")
print(f"  MAE:  {np.mean(cv_scores['mae']):.2f} ± {np.std(cv_scores['mae']):.2f}")
print(f"  MAPE: {np.mean(cv_scores['mape']):.2f}% ± {np.std(cv_scores['mape']):.2f}%")

# %% [markdown]
# ## 13.4.2 Ekstrapolasyon Sorununa Çözüm: Fark Üzerinden Tahmin
# Hedef seviye yerine aylık değişim; tahmin `lag_1` ile toplanarak seviyeye döner.

# %%
# =============================================================
# 9) EKSTRAPOLASYON SORUNUNA ÇÖZÜM: FARK ÜZERİNDEN TAHMİN
# =============================================================
#
# Seviyeyi (Passengers) değil, bir önceki aya göre DEĞİŞİMİ
# tahmin edelim. Değişimin aralığı eğitim ve test döneminde
# benzer olduğundan ağaçlar bu hedefte sınır sorunu yaşamaz.
#
#   hedef:          d(t) = y(t) - y(t-1)
#   geri dönüşüm:   ŷ(t) = y(t-1) + d̂(t)      (y(t-1) = lag_1)

d_train = y_train - X_train['lag_1']

diff_model = xgb.XGBRegressor(n_estimators=best_n, **params)
diff_model.fit(X_train, d_train)

y_test_pred_diff = X_test['lag_1'].values + diff_model.predict(X_test)

print("\n" + "=" * 50)
print("SEVİYE MODELİ vs FARK MODELİ (Test)")
print("=" * 50)
calculate_metrics(y_test, y_test_pred, "Seviye modeli")
calculate_metrics(y_test, y_test_pred_diff, "Fark modeli")
print(f"\nFark modelinin testteki en büyük tahmini: {y_test_pred_diff.max():.0f}")
