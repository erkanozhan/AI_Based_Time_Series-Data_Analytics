# %% [markdown]
# # Bölüm 10 — VAR: Çok Değişkenli Zaman Serisi Modeli (statsmodels)
# Ders notu: Course_notes.md, Bölüm 10.9. Bu dosya bölümdeki Python VAR uygulamasının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch10_var.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.
#
# Veri: `data/macro.csv` — aylık üç seri: `inflation` (enflasyon, %), `interest` (politika faizi, %),
# `exchange` (döviz kuru, USD/TRY). Kendi verinizle değiştirebilirsiniz.
# Çıktıların nasıl yorumlanacağı için notta Bölüm 10.9.1'deki tabloya bakın.

# %%
# statsmodels Colab'da kurulu gelir; ek kurulum gerekmez.
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

DATA_URL = "https://raw.githubusercontent.com/erkanozhan/AI_Based_Time_Series-Data_Analytics/main/data/"


def veri_yolu(dosya):
    """Veriyi önce yerel data/ klasöründe arar; bulamazsa GitHub'dan okur (Colab için)."""
    for aday in (Path("data") / dosya, Path("../data") / dosya, Path("../../data") / dosya):
        if aday.exists():
            return str(aday)
    return DATA_URL + dosya


# %% [markdown]
# ## 1) Veri setini okuma ve temel hazırlık (Bölüm 10.3)
# Üç makroekonomik değişken birbirini etkiler: merkez bankası enflasyonu kontrol etmek için faizi
# artırabilir, faiz artışı döviz kurunu, döviz kuru da ithal mallar üzerinden enflasyonu etkileyebilir.
# VAR modeli bu karşılıklı etkileşimleri yakalamaya çalışır. VAR eksik veri kaldırmadığı için
# `dropna()` ile eksik satırlar çıkarılır; `asfreq("MS")` ise verinin aylık (ay başı) olduğunu
# pandas'a açıkça söyler (statsmodels'ın "frekans bilgisi yok" uyarısını da önler).

# %%
df = pd.read_csv(veri_yolu("macro.csv"), parse_dates=["date"], index_col="date")

vars_selected = ["inflation", "interest", "exchange"]
df_var = df[vars_selected].dropna().asfreq("MS")

print("Veri setinin son gözlemleri:")
print(df_var.tail())
print(f"\nToplam gözlem sayısı: {len(df_var)}")

# Gözlem sayısı önemli: 3 değişken ve 4 gecikmede her denklemde
# 3 × 4 = 12 katsayı + 1 sabit = 13 parametre, toplam 3 × 13 = 39 parametre olur.

# %% [markdown]
# ## 2) Serilerin görselleştirilmesi
# Belirgin trend, yapısal kırılma (ani değişim) ve serilerin birlikte hareket edip etmediğine bakın.
# Türkiye verisinde 2018 ve 2021–2022 dönemlerindeki sert hareketler kriz dönemlerine karşılık gelir.

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, col in enumerate(vars_selected):
    axes[i].plot(df_var.index, df_var[col], linewidth=1.2)
    axes[i].set_ylabel(col)
    axes[i].grid(True, alpha=0.3)

axes[0].set_title("Değişkenlerin Zaman İçindeki Seyri")
axes[2].set_xlabel("Tarih")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3) Durağanlık testi — ADF (Bölüm 10.3.3)
# Durağan olmayan serilerle sahte (spurious) ilişkiler bulunabilir.
# ADF testinde H0: seri durağan değildir (birim kök vardır); p < 0,05 ise H0 reddedilir.

# %%
def adf_test(series, name):
    """
    ADF testi uygular ve sonuçları yorumlar.

    Test istatistiği kritik değerlerden küçükse (daha negatifse)
    veya p-değeri 0.05'ten küçükse seri durağan kabul edilir.
    """
    # statsmodels 0.15 bu satırda bir FutureWarning gösterebilir; zararsızdır.
    result = adfuller(series, autolag="AIC")

    # Şimdiki sürümlerde adfuller bir demet (tuple) döndürür:
    # [0]: test istatistiği, [1]: p-değeri, [2]: kullanılan gecikme,
    # [3]: gözlem sayısı, [4]: kritik değerler (sözlük)
    # İleriki sürümler adlandırılmış bir sonuç nesnesi döndürecek; iki durumu da destekliyoruz.
    if isinstance(result, tuple):
        test_stat, p_value, used_lag, _, critical_values = result[:5]
    else:
        test_stat, p_value = result.statistic, result.pvalue
        used_lag, critical_values = result.lags, result.critical_values

    print(f"\n{name}:")
    print(f"  Test istatistiği : {test_stat:.4f}")
    print(f"  p-değeri         : {p_value:.4f}")
    print(f"  Kullanılan gecikme: {used_lag}")
    print(f"  Kritik değerler  : %1: {critical_values['1%']:.3f}, "
          f"%5: {critical_values['5%']:.3f}, "
          f"%10: {critical_values['10%']:.3f}")

    if p_value < 0.05:
        print("  → Seri durağan görünüyor (H0 reddedildi)")
    else:
        print("  → Seri muhtemelen durağan değil (H0 reddedilemedi)")
        print("    Fark almak gerekebilir.")


print("=" * 55)
print("DURAĞANLIK TESTLERİ (ADF)")
print("=" * 55)

for col in vars_selected:
    adf_test(df_var[col], col)

# Durağan olmayan seriler için en yaygın çözüm birinci farktır:
#   df_var["inflation_d"] = df_var["inflation"].diff()
# Fark alınınca ilk gözlem NaN olur, dropna() ile temizlenir ve ADF tekrarlanır.
# Bu örnekte eğitim amaçlı orijinal serilerle devam ediyoruz.
# Gerçek bir çalışmada durağan olmayan seriler mutlaka dönüştürülmelidir.

# %% [markdown]
# ## 4) VAR modelinin kurulması ve gecikme seçimi (Bölüm 10.4)
# Bilgi kriterleri uyum ile parametre sayısı arasında denge kurar; düşük değer daha iyidir.
# AIC daha esnektir (fazla gecikmeye izin verebilir), BIC tutucudur, HQIC ikisinin arasındadır.

# %%
model = VAR(df_var)

# maxlags=8: 1'den 8'e kadar tüm gecikmeler için AIC, BIC, FPE, HQIC hesaplanır.
lag_order_results = model.select_order(maxlags=8)

print("\n" + "=" * 55)
print("GECİKME SEÇİMİ")
print("=" * 55)
print(lag_order_results.summary())

print("\nKriterlere göre önerilen gecikmeler:")
print(f"  AIC : {lag_order_results.selected_orders['aic']}")
print(f"  BIC : {lag_order_results.selected_orders['bic']}")
print(f"  HQIC: {lag_order_results.selected_orders['hqic']}")

# Genel kural: öngörü amaçlıysa AIC, tutumlu model isteniyorsa BIC.
# Bu veride AIC 7 gecikme önerir; 3 değişkenli VAR(7) denklem başına 22 parametre
# demektir ve model stabil çıkmaz. Bu yüzden tutucu BIC'nin önerisini kullanıyoruz.
selected_lag = lag_order_results.selected_orders['bic']
# Kriter 0 gecikme önerirse VAR kurulamaz; en az 1 gecikme kullanıyoruz.
selected_lag = max(1, selected_lag)
print(f"\nSeçilen gecikme (BIC'ye göre): {selected_lag}")

# %%
# fit(): her denklem OLS ile ayrı ayrı tahmin edilir. VAR'da tüm denklemler
# aynı açıklayıcı değişkenlere sahip olduğundan bu, sistemi birlikte
# tahmin etmekle aynı sonucu verir.
results = model.fit(selected_lag)
# Kısa yol: model.fit(maxlags=8, ic="bic") gecikmeyi BIC ile seçip modeli tek adımda kurar
# (bu veride yine VAR(2) çıkar).

print("\n" + "=" * 55)
print("MODEL TAHMİN SONUÇLARI")
print("=" * 55)
print(results.summary())

# Özet tabloda her denklem için katsayılar (const, L1.inflation, ...), standart
# hatalar, t-istatistikleri ve p-değerleri; en altta artıkların korelasyon matrisi
# bulunur. VAR katsayılarını tek tek yorumlamak zordur; IRF ve FEVD yorumu kolaylaştırır.

# %% [markdown]
# ## 5) Stabilite kontrolü (Bölüm 10.5)
# Sisteme verilen bir şokun etkisi zamanla sönmeli, patlamamalıdır. Stabil olmayan VAR ile
# IRF, tahmin ve FEVD güvenilmez olur.

# %%
print("\n" + "=" * 55)
print("STABİLİTE KONTROLÜ")
print("=" * 55)

is_stable = results.is_stable()
print(f"Model stabil mi? {is_stable}")

if is_stable:
    print("Tüm öz değerler birim çemberin içinde - model stabil.")
else:
    print("UYARI: Köklerden bazıları birim çember dışında!")
    print("Model yeniden gözden geçirilmeli:")
    print("  - Gecikme sayısı değiştirilebilir")
    print("  - Seriler fark alınarak durağanlaştırılabilir")
    print("  - Aykırı gözlemler incelenebilir")

# Alıştırma: 4. adımda AIC'nin önerdiği gecikmeyi kullanın
#   selected_lag = lag_order_results.selected_orders['aic']
# ve stabilite sonucunu karşılaştırın. Bu veride VAR(7) stabil çıkmaz: seriler durağan
# olmadığı için bir öz değerin modülü 1'i aşar (yaklaşık 1,024).
# Serilerin farkını alarak da deneyin (1. adımdan sonra):
#   df_var = df_var.diff().dropna()
# Farkı alınmış serilerde BIC 1 gecikme önerir, en büyük |öz değer| yaklaşık 0,55'e iner.

# DİKKAT: statsmodels'ta results.roots, karakteristik polinomun
# köklerini verir; bunlar eşlik (companion) matrisinin öz değerlerinin
# TERSİDİR. Stabilite için kökler birim çemberin DIŞINDA (|kök| > 1),
# öz değerler ise İÇİNDE (|öz değer| < 1) olmalıdır. İkisi aynı koşuldur.
roots = results.roots
print("\nKarakteristik kökler ve karşılık gelen öz değerler (mutlak değer):")
for i, root in enumerate(roots):
    print(f"  Kök {i+1}: |kök| = {np.abs(root):.4f}  ->  |öz değer| = {1 / np.abs(root):.4f}")
print("(Tüm |kök| değerleri 1'den büyük, yani tüm |öz değer| değerleri 1'den küçük olmalı)")

# %% [markdown]
# ## 6) Artık (residual) analizi
# İyi bir modelde artıkların ortalaması sıfır, otokorelasyonu yok ve varyansı sabittir.
# Durbin-Watson (DW) birinci derece otokorelasyonu ölçer: DW ≈ 2 ideal, 1,5–2,5 arası genellikle kabul edilebilir.

# %%
print("\n" + "=" * 55)
print("ARTIK ANALİZİ")
print("=" * 55)

residuals = results.resid
dw_stats = durbin_watson(residuals)

print("\nDurbin-Watson istatistikleri:")
for i, col in enumerate(vars_selected):
    dw = dw_stats[i]
    if 1.5 <= dw <= 2.5:
        yorum = "kabul edilebilir"
    elif dw < 1.5:
        yorum = "pozitif otokorelasyon olabilir"
    else:
        yorum = "negatif otokorelasyon olabilir"
    print(f"  {col}: {dw:.3f} ({yorum})")

print("\n  Not: 2'ye yakın değerler otokorelasyon olmadığını gösterir.")

# DW yalnızca bir önceki ayla ilişkiye bakar. Portmanteau (Ljung-Box tipi) testi ise
# 12 gecikmeye kadar tüm denklemlerin artıklarını birlikte sınar. H0: otokorelasyon yok.
whiteness = results.test_whiteness(nlags=12)
print(f"\nPortmanteau testi (12 gecikme): p-değeri = {whiteness.pvalue:.4f}")
if whiteness.pvalue < 0.05:
    print("  → Artıklarda daha uzun gecikmelerde otokorelasyon kalmış (H0 reddedildi).")
    print("    Düzey serilerle kurulan modelin bir eksikliği; fark almak ya da gecikmeyi artırmak denenebilir.")

# Artıklar sıfır çizgisi etrafında rastgele dağılmalı; trend, periyodik örüntü
# veya değişen varyans model sorunlarına işaret eder.
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for i, col in enumerate(vars_selected):
    axes[i].plot(residuals.index, residuals[col], linewidth=0.8)
    axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[i].set_ylabel(col)
    axes[i].grid(True, alpha=0.3)

axes[0].set_title("Model Artıkları")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7) Kısa dönem tahmin (forecast)
# Tahmin için son `p` gözlem (p = gecikme sayısı) başlangıç noktası olarak verilir.
# VAR tahminleri kısa vadede makuldür; ufuk uzadıkça belirsizlik hızla artar.

# %%
print("\n" + "=" * 55)
print("TAHMİN (FORECAST)")
print("=" * 55)

forecast_horizon = 4  # 4 dönem (ay) ileriye tahmin

# Son 'selected_lag' gözlemi başlangıç değeri olarak alıyoruz
lagged_values = df_var.values[-selected_lag:]
forecast_values = results.forecast(y=lagged_values, steps=forecast_horizon)

# pd.infer_freq() bazen None dönebilir, bu durumu ele alıyoruz
freq = pd.infer_freq(df_var.index)
if freq is None:
    freq = 'MS'  # Month Start - ay başı
    print(f"Frekans otomatik belirlenemedi, '{freq}' varsayıldı.")

idx_forecast = pd.date_range(
    start=df_var.index[-1] + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq=freq
)

df_forecast = pd.DataFrame(forecast_values, index=idx_forecast, columns=vars_selected)

print(f"\n{forecast_horizon} dönemlik tahminler:")
print(df_forecast.round(2))

# %%
# Son 24 aylık gerçek değerler ve tahminler
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, col in enumerate(vars_selected):
    axes[i].plot(df_var.index[-24:], df_var[col].iloc[-24:],
                 label='Gerçek', linewidth=1.2)
    axes[i].plot(df_forecast.index, df_forecast[col],
                 'r--', label='Tahmin', linewidth=1.2, marker='o')
    axes[i].set_ylabel(col)
    axes[i].legend(loc='upper left')
    axes[i].grid(True, alpha=0.3)

axes[0].set_title("Gerçek Değerler ve Tahminler")
plt.tight_layout()
plt.show()

# Tahminler mevcut trendin devamı gibi görünmeli; çok keskin dönüşler veya
# mantıksız değerler (negatif enflasyon gibi) model sorunlarına işaret edebilir.

# %% [markdown]
# ## 8) Etki-tepki analizi — IRF (Bölüm 10.7)
# "Bir değişkene verilen şokun diğer değişkenler üzerindeki etkisi zamanla nasıl gelişir?"
# Yatay eksen şoktan sonra geçen dönem, dikey eksen tepkinin büyüklüğüdür; geniş güven bandı
# o dönemdeki tepkinin istatistiksel olarak belirsiz olduğunu gösterir.
# `orth=False`: bir birimlik (ortogonal olmayan) şoklar; `orth=True` Cholesky şokları (sütun sırasına bağlı).

# %%
print("\n" + "=" * 55)
print("ETKİ-TEPKİ ANALİZİ (IRF)")
print("=" * 55)

# 12 dönemlik (1 yıl) tepkiler
irf = results.irf(12)

# Tüm değişken çiftleri: her satır bir şoku, her sütun o şoka verilen tepkiyi gösterir
fig_irf = irf.plot(orth=False)
plt.suptitle("Impulse Response Functions", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Faiz şoku enflasyonu nasıl etkiler? Teoriye göre faiz artışı enflasyonu düşürmeli.
# Bu veride tepki negatif çıkar, ancak güven bandı her dönemde sıfırı içerir
# (Granger testindeki yüksek p-değeriyle uyumlu).
fig_irf_pair = irf.plot(impulse="interest", response="inflation")
plt.suptitle("Faiz Şokuna Enflasyonun Tepkisi")
plt.tight_layout()
plt.show()

# Döviz kuru şoku enflasyonu nasıl etkiler? TL'nin değer kaybı ithal malları
# pahalılaştırarak enflasyonu artırmalı ("exchange rate pass-through").
# Bu veride kurdaki 1 birimlik artış enflasyonu bir ay sonra yaklaşık 3,7 puan, 4. ayda
# yaklaşık 9 puan artırır; tepki sonra yavaşça azalır ve bant sıfırın üstünde kalır.
fig_irf_exc = irf.plot(impulse="exchange", response="inflation")
plt.suptitle("Döviz Kuru Şokuna Enflasyonun Tepkisi")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9) Tahmin hatası varyans ayrıştırması — FEVD (Bölüm 10.8)
# Bir değişkenin tahmin hatasının ne kadarı kendi şoklarından, ne kadarı diğer değişkenlerin
# şoklarından kaynaklanıyor? Tabloda satırlar dönemler, sütunlar katkı paylarıdır (her satırın toplamı 1).

# %%
print("\n" + "=" * 55)
print("VARYANS AYRIŞTIRMASI (FEVD)")
print("=" * 55)

fevd = results.fevd(12)  # 12 dönemlik ufuk
fevd.summary()  # summary() tabloyu kendisi yazdırır (None döndürür; print'e sarmayın)

# Her değişken için ayrı grafik; renkli alanlar her şokun katkı payıdır.
fig_fevd = fevd.plot()
plt.suptitle("Forecast Error Variance Decomposition", y=1.02)
plt.tight_layout()
plt.show()

# İlk dönemlerde değişken genellikle kendi şoklarından etkilenir; ufuk uzadıkça
# diğer değişkenlerin etkisi belirginleşir ve paylar sabitlenir.

# %% [markdown]
# ## 10) Granger nedensellik testleri (Bölüm 10.6)
# "X, Y'nin Granger nedenidir" = X'in geçmiş değerleri Y'nin tahminini iyileştirir
# (gerçek neden-sonuç ilişkisi olduğu anlamına gelmez).
# H0: X'in geçmiş değerleri Y denklemine ek bilgi katmıyor; p < 0,05 ise H0 reddedilir.

# %%
print("\n" + "=" * 55)
print("GRANGER NEDENSELLİK TESTLERİ")
print("=" * 55)

# Test 1: Faiz → Enflasyon
gc_int_inf = results.test_causality(
    caused="inflation",      # Etkilenen (bağımlı) değişken
    causing=["interest"],    # Etkileyen (açıklayıcı) değişken
    kind="f"                 # F-testi kullan
)
print("\n1) Faiz → Enflasyon:")
print(gc_int_inf.summary())

# Test 2: Döviz kuru → Enflasyon
gc_exc_inf = results.test_causality(
    caused="inflation",
    causing=["exchange"],
    kind="f"
)
print("\n2) Döviz Kuru → Enflasyon:")
print(gc_exc_inf.summary())

# Test 3: Enflasyon → Faiz (merkez bankası enflasyona tepki veriyor mu?)
gc_inf_int = results.test_causality(
    caused="interest",
    causing=["inflation"],
    kind="f"
)
print("\n3) Enflasyon → Faiz:")
print(gc_inf_int.summary())

# Test 4: Döviz kuru → Faiz (merkez bankası döviz kuruna tepki veriyor mu?)
gc_exc_int = results.test_causality(
    caused="interest",
    causing=["exchange"],
    kind="f"
)
print("\n4) Döviz Kuru → Faiz:")
print(gc_exc_int.summary())

# Çift yönlü nedensellik de mümkündür (enflasyon faizi, faiz de enflasyonu etkiler);
# bu tür karşılıklı etkileşimler VAR modelinin varlık sebebidir.

# %% [markdown]
# ## Özet

# %%
print("\n" + "=" * 55)
print("ANALİZ TAMAMLANDI")
print("=" * 55)
print("""
Bu VAR analizinde şunları yaptık:

1. Verileri hazırladık ve görselleştirdik
2. Durağanlığı ADF testi ile kontrol ettik
3. Bilgi kriterleriyle optimal gecikme sayısını belirledik
4. Modeli tahmin ettik ve stabilitesini kontrol ettik
5. Artıkları inceleyerek model uyumunu değerlendirdik
6. Kısa dönem tahminler ürettik
7. IRF ile şokların yayılımını analiz ettik
8. FEVD ile varyans kaynaklarını ayrıştırdık
9. Granger nedensellik testleri ile öngörü ilişkilerini inceledik

Unutulmaması gerekenler:
- VAR sonuçları sadece korelasyon/öngörü ilişkilerini gösterir,
  gerçek nedensellik için ek analizler gerekir.
- Durağan olmayan serilerle çalışmak sahte ilişkilere yol açabilir.
- Yapısal kırılmalar (kriz dönemleri) model performansını etkiler.
- Kısa dönem tahminler uzun döneme göre daha güvenilirdir.
""")
