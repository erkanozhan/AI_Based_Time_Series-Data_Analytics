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

# ---- 3.2 Durağan ve Durağan Olmayan: beyaz gürültü ve rastgele yürüyüş ----
set.seed(42)
beyaz_gurultu <- rnorm(200)             # durağan
rastgele_yuruyus <- cumsum(rnorm(200))  # durağan değil (birim kök)

par(mfrow = c(1, 2))
plot.ts(beyaz_gurultu, main = "Beyaz Gürültü (durağan)")
plot.ts(rastgele_yuruyus, main = "Rastgele Yürüyüş (durağan değil)")
par(mfrow = c(1, 1))

# ---- 3.2 ADF testi ----
# p-değeri 0.01 ile 0.10 arasına sıkıştırıldığı için "p-value smaller/greater than
# printed p-value" uyarısı görebilirsiniz; bu bir hata değildir.
print(adf.test(beyaz_gurultu))           # küçük p-değeri -> H0 reddedilir -> durağan
print(adf.test(rastgele_yuruyus))        # büyük p-değeri -> birim kök var
print(adf.test(diff(rastgele_yuruyus)))  # farkı alınınca durağanlaşır
