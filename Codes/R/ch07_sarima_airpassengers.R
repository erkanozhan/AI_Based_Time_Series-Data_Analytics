# =============================================================================
# Bölüm 7 — Klasik İstatistiksel Modeller: ARIMA ve SARIMA
# Ders notu: Course_notes.md, Bölüm 7.6 (R Uygulaması: AirPassengers ile SARIMA)
#
# Çalıştırma: depo kök dizininden  Rscript Codes/R/ch07_sarima_airpassengers.R
#             ya da RStudio'da dosyayı açıp satır satır (Ctrl+Enter) çalıştırın.
# Gerekli paketler: forecast, tseries
#   (Kurulum için bir kez: source("Codes/install_packages.R"))
#
# Not: "#>" ile başlayan yorumlar, notta verilen beklenen çıktılardır.
# =============================================================================

library(forecast)
library(tseries)

# ---- 7.6.1 Veriyi Görselleştirme ----
# Veriyi yükle ve çiz
data(AirPassengers)
plot(AirPassengers, main = "AirPassengers Verisi: Trend ve Artan Varyans",
     ylab = "Yolcu Sayısı", xlab = "Yıl", col = "darkblue")

# ---- 7.6.2 Durağanlık Testleri ----
print(adf.test(AirPassengers))
#> Dickey-Fuller = -7.3186, Lag order = 5, p-value = 0.01

print(kpss.test(AirPassengers))
#> KPSS Level = 2.7395, Truncation lag parameter = 4, p-value = 0.01

# ---- 7.6.3 Seriyi Durağanlaştırma ----
# Önce log dönüşümü, sonra mevsimsel (lag = 12) ve normal (lag = 1) fark
AP_stationary <- diff(diff(log(AirPassengers), lag = 12))

plot(AP_stationary, main = "Dönüştürülmüş AirPassengers Serisi",
     ylab = "Fark Değerleri", col = "darkblue")
abline(h = 0, lty = 2)

# Testleri tekrarlayalım
print(adf.test(AP_stationary))
#> Dickey-Fuller = -5.1993, Lag order = 5, p-value = 0.01

print(kpss.test(AP_stationary))
#> KPSS Level = 0.084365, Truncation lag parameter = 4, p-value = 0.1

# forecast paketi gereken fark sayılarını doğrudan da önerebilir
nsdiffs(log(AirPassengers))                  # gereken mevsimsel fark sayısı (D)
#> [1] 1
ndiffs(diff(log(AirPassengers), lag = 12))   # mevsimsel farktan sonra gereken normal fark sayısı (d)
#> [1] 1

# ---- 7.6.4 Model Belirleme (ACF ve PACF) ----
par(mfrow = c(1, 2))  # grafikleri yan yana göster
acf(AP_stationary, lag.max = 36, main = "ACF")
pacf(AP_stationary, lag.max = 36, main = "PACF")
par(mfrow = c(1, 1))

# Yorumda geçen değerler: lag-1 ve lag-12 ACF (yaklaşık -0.34 ve -0.39)
acf_ap <- acf(as.numeric(AP_stationary), lag.max = 36, plot = FALSE)
print(round(acf_ap$acf[c(1, 12) + 1], 2))

# ---- 7.6.5 Model Kurma: auto.arima() ve Seçilen Modelin Yorumu ----
fit <- auto.arima(log(AirPassengers), seasonal = TRUE)
print(fit)
#> ARIMA(0,1,1)(0,1,1)[12]
#>           ma1     sma1
#>       -0.4018  -0.5569
#> s.e.   0.0896   0.0731
#> sigma^2 = 0.001371:  log likelihood = 244.7
#> AIC=-483.4   AICc=-483.21   BIC=-474.77

# ---- 7.6.6 Teşhis: Artıkların İncelenmesi ----
checkresiduals(fit)
#> Q* = 26.446, df = 22, p-value = 0.233

# ---- 7.6.7 Tahmin (Öngörü) ----
# Gelecek 24 ay için tahmin (log ölçekte)
fc <- forecast(fit, h = 24)
plot(fc, main = "Gelecek 24 Ay için log(Yolcu Sayısı) Tahmini")
grid()

# Orijinal ölçeğe dönmek için exp() uygularız
round(exp(fc$mean[c(1, 12, 24)]), 1)   # 1., 12. ve 24. ay tahminleri
#> [1] 450.4 477.2 525.5
