# =============================================================================
# Bölüm 6 — Veri Manipülasyonu, Görselleştirme ve ACF/PACF
# Ders notu: Course_notes.md, Bölüm 6 (plot, ggplot2, aggregate, lag, decompose,
#            ACF, PACF, USgas korelogramları, AR/MA imzaları)
#
# Çalıştırma: depo kök dizininden  Rscript Codes/R/ch06_manipulasyon_acf.R
#             ya da RStudio'da dosyayı açıp satır satır (Ctrl+Enter) çalıştırın.
# Gerekli paketler: TSstudio (USgas verisi), ggplot2
#   (Kurulum için bir kez: source("Codes/install_packages.R"))
# =============================================================================

library(TSstudio)
library(ggplot2)

data(USgas)

# ---- 6.1.1 plot() ile Hızlı Grafik ----
plot(USgas,
     main = "ABD Doğal Gaz Tüketimi (2000-2019)",
     ylab = "Milyar Kübik Fit",
     xlab = "Yıl",
     col = "blue")
grid()

# ---- 6.1.2 Gelişmiş Görselleştirme: ggplot2 ----
# USgas ts nesnesini data.frame'e dönüştür
df_gg <- data.frame(
  tarih = seq(as.Date("2000-01-01"), by = "month", length.out = length(USgas)), # Aylık tarih dizisi
  deger = as.numeric(USgas)                                                     # ts değerlerini sayısal vektöre çevir
)

# Zaman serisi grafiği ve yumuşatılmış trend çizgisi
# (Rscript ile çalıştırırken ggplot nesnesinin çizilmesi için print() kullanıyoruz)
p <- ggplot(df_gg, aes(x = tarih, y = deger)) +
  geom_line(color = "blue", linewidth = 0.8) +
  geom_smooth(method = "loess", formula = y ~ x,
              color = "red", se = FALSE, linetype = "dashed") + # Trend çizgisi ekle
  labs(title = "ABD Doğal Gaz Tüketimi (2000-2019)",
       subtitle = "ggplot2 ile Gelişmiş Görselleştirme",
       x = "Tarih",
       y = "Milyar Kübik Fit") +
  theme_minimal()
print(p)

# ---- 6.2.1 Frekansı Düşürmek: aggregate() ----
# Aylık veriyi yıllık toplam tüketime çevirelim
USgas_yillik <- aggregate(USgas, nfrequency = 1, FUN = sum)
print(USgas_yillik)

# ---- 6.2.2 Gecikmeli Değerler: lag() ----
# 1 ay önceki değeri (lag-1) ve 12 ay önceki değeri (lag-12) oluşturalım
USgas_lag1  <- stats::lag(USgas, k = -1)
USgas_lag12 <- stats::lag(USgas, k = -12)

# Orijinal seri ile gecikmeli değerleri yan yana koyalım
comparison_df <- cbind(
    Original = USgas,
    Lag1  = USgas_lag1,
    Lag12 = USgas_lag12
)
print(head(comparison_df, 15)) # İlk 15 satır kaydırmayı açıkça gösterir

# ---- 6.2.3 Bileşenlere Ayırma: decompose() ----
# USgas serisini bileşenlerine ayıralım (toplamsal model)
USgas_ayristir <- decompose(USgas)
plot(USgas_ayristir)

# Her ayın mevsimsel etkisi (12 değer)
print(round(USgas_ayristir$figure))

# ---- 6.3.3 Elle Hesaplama Örneği ----
# x = [10, 12, 15, 11, 17] için lag-1 otokorelasyonu (elle: -11/34 = -0.324)
print(acf(c(10, 12, 15, 11, 17), plot = FALSE)$acf[2])

# ---- 6.3.4 Python ve R ile ACF ----
data <- c(20, 22, 21, 23, 24)          # Örnek bir zaman serisi vektörü oluştur
acf_result <- acf(data, plot = FALSE)  # Grafik çizmeden ACF değerlerini hesapla
# R'da acf() çıktısının ilk elemanı lag-0'dır (her zaman 1), bu yüzden lag-1 için 2. elemanı alırız.
cat("Lag-1 ACF:", round(acf_result$acf[2], 3), "\n")

# ---- 6.4.3 Python ve R ile PACF ----
data <- c(20, 22, 21, 23, 24)
pacf_result <- pacf(data, plot = FALSE)
# Dikkat: pacf() çıktısı lag-1'den başlar (lag-0 yoktur), bu yüzden lag-2 için 2. elemanı alırız.
cat("Lag-2 PACF:", round(pacf_result$acf[2], 3), "\n")

# ---- 6.5 Uygulama: USgas Serisinin ACF ve PACF Grafikleri ----
# USgas verisinin ACF ve PACF grafiklerini alt alta çizelim
par(mfrow = c(2, 1))  # 2 satır, 1 sütunluk grafik düzeni
acf(USgas,  lag.max = 36, main = "Otokorelasyon Fonksiyonu (ACF)")
pacf(USgas, lag.max = 36, main = "Kısmi Otokorelasyon Fonksiyonu (PACF)")
par(mfrow = c(1, 1))  # Grafik düzenini eski hâline getir

# Güven sınırı: 1.96 / sqrt(T)
print(1.96 / sqrt(length(USgas)))

# Yorumda geçen sayısal değerler (eksende gecikme sayısını görmek için as.numeric)
acf_us  <- acf(as.numeric(USgas),  lag.max = 36, plot = FALSE)
pacf_us <- pacf(as.numeric(USgas), lag.max = 36, plot = FALSE)
print(round(acf_us$acf[c(1, 2, 12, 24) + 1], 2))   # lag 1, 2, 12, 24
print(round(pacf_us$acf[c(1, 2, 13)], 2))          # lag 1, 2, 13

# ---- 6.6.4 İmzaları Yan Yana Görmek ----
set.seed(42)
wn  <- rnorm(400)                                       # Beyaz gürültü
ar1 <- arima.sim(model = list(ar = 0.7), n = 400)       # AR(1), phi = 0.7
ma1 <- arima.sim(model = list(ma = 0.8), n = 400)       # MA(1), theta = 0.8

par(mfrow = c(3, 2))  # 3 satır (süreçler) x 2 sütun (ACF, PACF)
acf(wn,  lag.max = 15, main = "Beyaz gürültü: ACF");  pacf(wn,  lag.max = 15, main = "Beyaz gürültü: PACF")
acf(ar1, lag.max = 15, main = "AR(1): ACF");          pacf(ar1, lag.max = 15, main = "AR(1): PACF")
acf(ma1, lag.max = 15, main = "MA(1): ACF");          pacf(ma1, lag.max = 15, main = "MA(1): PACF")
par(mfrow = c(1, 1))

# Lag-1 örneklem otokorelasyonları (teorik: AR(1) 0.7, MA(1) 0.49)
cat("AR(1) lag-1 ACF:", round(acf(ar1, plot = FALSE)$acf[2], 2), "\n")
cat("MA(1) lag-1 ACF:", round(acf(ma1, plot = FALSE)$acf[2], 2), "\n")
