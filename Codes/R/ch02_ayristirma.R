# =============================================================================
# Bölüm 2 — Zaman Serisinin Temel Kavramları ve Bileşenleri
# Ders notu: Course_notes.md, Bölüm 2 (2.6 Mini Uygulama: AirPassengers ayrıştırma)
#
# Çalıştırma: depo kök dizininden  Rscript Codes/R/ch02_ayristirma.R
#             ya da RStudio'da dosyayı açıp satır satır (Ctrl+Enter) çalıştırın.
# Gerekli paketler: yok (yalnızca base R ve stats)
# =============================================================================

# ---- 2.6 Mini Uygulama: AirPassengers Serisini Ayrıştırmak ----
data("AirPassengers")

# 1) Çarpımsal ayrıştırma: xt = Tt x St x It
ayr_carp <- decompose(AirPassengers, type = "multiplicative")
plot(ayr_carp)

# Mevsimsel katsayılar (ortalaması 1): Temmuz ~1.23 -> trend seviyesinden ~%23 fazla
print(round(ayr_carp$figure, 3))
#>  [1] 0.910 0.884 1.007 0.976 0.981 1.113 1.227 1.220 1.060 0.922 0.801 0.899

# 2) Log dönüşümü + toplamsal ayrıştırma: log(xt) = log(Tt) + log(St) + log(It)
ayr_log <- decompose(log(AirPassengers), type = "additive")
plot(ayr_log)

# 3) Daha modern ve sağlam bir yöntem: STL (Seasonal-Trend decomposition using Loess)
ayr_stl <- stl(log(AirPassengers), s.window = "periodic")
plot(ayr_stl)
