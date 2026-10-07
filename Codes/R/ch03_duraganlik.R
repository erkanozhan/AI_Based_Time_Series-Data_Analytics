# =============================================================================
# Bölüm 3 — Zaman Serisi Tipleri
# Ders notu: Course_notes.md, Bölüm 3 (3.2 Durağan ve Durağan Olmayan: mini uygulama)
#
# Çalıştırma: depo kök dizininden  Rscript Codes/R/ch03_duraganlik.R
#             ya da RStudio'da dosyayı açıp satır satır (Ctrl+Enter) çalıştırın.
# Gerekli paketler: tseries
#   (Kurulum için bir kez: source("Codes/install_packages.R"))
# =============================================================================

library(tseries)

# ---- 3.2 Karakteristik denklemin kökleri: polyroot() ----
# Katsayılar sabit terimden başlayarak verilir: c(1, -0.5) -> 1 - 0.5z
# Mod() her kökün sıfıra uzaklığını (mutlak değerini) verir.
print(Mod(polyroot(c(1, -0.5))))        # AR(1), phi = 0.5
#> [1] 2
print(Mod(polyroot(c(1, -1))))          # rastgele yürüyüş, phi = 1 -> birim kök
#> [1] 1
print(Mod(polyroot(c(1, -1.5, 0.5))))   # 1 - 1.5z + 0.5z^2 = 0
#> [1] 1 2

# ---- 3.2 Durağan ve Durağan Olmayan: beyaz gürültü ve rastgele yürüyüş ----
set.seed(42)
beyaz_gurultu <- rnorm(200)             # durağan
rastgele_yuruyus <- cumsum(rnorm(200))  # durağan değil (birim kök)

par(mfrow = c(1, 2))
plot.ts(beyaz_gurultu, main = "Beyaz Gürültü (durağan)")
plot.ts(rastgele_yuruyus, main = "Rastgele Yürüyüş (durağan değil)")
par(mfrow = c(1, 1))

# ---- 3.2 ADF testi ----
# H0: birim kök var (seri durağan değil), H1: seri durağan.
# adf.test() p-değerini bir tablodan okuduğu için 0.01 ile 0.99 arasına
# sıkıştırır; tablonun dışında "p-value smaller/greater than printed p-value"
# uyarısı görebilirsiniz. Bu bir hata değildir.
print(adf.test(beyaz_gurultu))           # küçük p-değeri -> H0 reddedilir -> durağan
#> Dickey-Fuller = -5.6389, Lag order = 5, p-value = 0.01
print(adf.test(rastgele_yuruyus))        # büyük p-değeri -> H0 reddedilemez (birim kök)
#> Dickey-Fuller = -1.9193, Lag order = 5, p-value = 0.6099
print(adf.test(diff(rastgele_yuruyus)))  # farkı alınınca durağanlaşır
#> Dickey-Fuller = -6.09, Lag order = 5, p-value = 0.01
