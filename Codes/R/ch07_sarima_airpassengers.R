# =============================================================================
# Bölüm 7 — Klasik İstatistiksel Modeller: ARIMA, SARIMA ve Üstel Düzeltme
# Ders notu: notlar/07_klasik_istatistiksel_modeller_arima_ve_sarima.md, Bölüm 7.2 (karakteristik kökler), 7.6 (R Uygulaması: AirPassengers ile SARIMA) ve 7.8 (ETS ve Theta)
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

# ---- 7.2 Karakteristik denklemin kökleri ----
# Katsayılar küçük üsten büyüğe: 1 - 0.6z - 0.16z^2 = 0
Mod(polyroot(c(1, -0.6, -0.16)))        # köklerin mutlak değerleri
#> [1] 1.25 5.00

# ---- 7.6.1 Veriyi Görselleştirme ----
# Veriyi yükle ve çiz
data(AirPassengers)
plot(AirPassengers, main = "AirPassengers Verisi: Trend ve Artan Varyans",
     ylab = "Yolcu Sayısı", xlab = "Yıl", col = "darkblue")

# ---- 7.6.2 Durağanlık Testleri ----
print(adf.test(AirPassengers))
#> Dickey-Fuller = -7.3186, Lag order = 5, p-value = 0.01

print(adf.test(AirPassengers, k = 13))   # 12 aylık hafızayı kapsayan gecikme sayısıyla
#> Dickey-Fuller = -2.1008, Lag order = 13, p-value = 0.5345

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

# Aynı test Box.test() ile: model artıklarında fitdf = kestirilen katsayı sayısı
Box.test(residuals(fit), lag = 24, type = "Ljung-Box", fitdf = 2)
#> X-squared = 26.446, df = 22, p-value = 0.233
Box.test(residuals(fit), lag = 24, type = "Ljung-Box")   # fitdf unutulursa
#> X-squared = 26.446, df = 24, p-value = 0.3309

# ---- 7.6.7 Tahmin (Öngörü) ----
# Gelecek 24 ay için tahmin (log ölçekte)
fc <- forecast(fit, h = 24)
plot(fc, main = "Gelecek 24 Ay için log(Yolcu Sayısı) Tahmini")
grid()

# Orijinal ölçeğe dönmek için exp() uygularız
round(exp(fc$mean[c(1, 12, 24)]), 1)   # 1., 12. ve 24. ay tahminleri
#> [1] 450.4 477.2 525.5

# ---- 7.8.1 Basit üstel düzeltme (SES): elle hesap ----
x <- c(40, 46, 43, 49)               # dört günlük satış
alfa <- 0.3
duzey <- x[1]                        # başlangıç düzeyi: ilk gözlem
for (t in 2:4) {
  duzey <- alfa * x[t] + (1 - alfa) * duzey   # yeni düzey = alfa * gözlem + (1 - alfa) * eski düzey
  cat(t, round(duzey, 3), "\n")
}
#> 2 41.8
#> 3 42.16
#> 4 44.212
round(alfa * (1 - alfa)^(0:5), 3)    # en yeni gözlemden geriye doğru ağırlıklar
#> [1] 0.300 0.210 0.147 0.103 0.072 0.050

# Aynı hesap forecast::ses() ile (alfa sabit, başlangıç düzeyi = ilk gözlem)
ses(x, alpha = 0.3, initial = "simple", h = 2)$mean
#> [1] 44.212 44.212

# ---- 7.8.6 AirPassengers: ETS, Holt-Winters ve Theta ----
# Test: 1960 (Bölüm 8.3 ile aynı ayrım). Modeller orijinal ölçekte, SARIMA log ölçekte kurulur.
egitim <- window(AirPassengers, end = c(1959, 12))   # 1949-1959 (132 ay)
test   <- window(AirPassengers, start = c(1960, 1))  # 1960 (12 ay)
h <- length(test)

fit_ets <- ets(egitim)               # E, T, S harflerini AICc ile kendisi seçer
print(fit_ets)
#> ETS(M,Ad,M)
#>     alpha = 0.758   beta = 0.0213   gamma = 1e-04   phi = 0.98
#>     AIC = 1244.458   AICc = 1250.511   BIC = 1296.348

fc_ets    <- forecast(fit_ets, h = h)
fc_hw     <- hw(egitim, seasonal = "multiplicative", h = h)   # Holt-Winters çarpımsal
fc_theta  <- thetaf(egitim, h = h)                            # Theta yöntemi
fc_sarima <- forecast(auto.arima(log(egitim)), h = h)         # havayolu modeli (log ölçek)
fc_snaive <- snaive(egitim, h = h)                            # mevsimsel naive

# Bölüm 8.3'teki metrik fonksiyonu
metrikler <- function(a, p) c(MAE  = mean(abs(a - p)),
                              RMSE = sqrt(mean((a - p)^2)),
                              MAPE = mean(abs((a - p) / a)) * 100)
a <- as.numeric(test)
tablo <- rbind(`Mevsimsel naive`        = metrikler(a, as.numeric(fc_snaive$mean)),
               `SARIMA (log)`           = metrikler(a, as.numeric(exp(fc_sarima$mean))),
               `ETS(M,Ad,M)`            = metrikler(a, as.numeric(fc_ets$mean)),
               `Holt-Winters çarpımsal` = metrikler(a, as.numeric(fc_hw$mean)),
               Theta                    = metrikler(a, as.numeric(fc_theta$mean)))
round(tablo, 2)
#>                          MAE  RMSE MAPE
#> Mevsimsel naive        47.83 50.71 9.99
#> SARIMA (log)           13.26 18.59 2.90
#> ETS(M,Ad,M)            22.80 27.40 4.66
#> Holt-Winters çarpımsal 11.13 16.58 2.37
#> Theta                  26.33 30.72 5.33

# Aylık hatalar (gerçek - tahmin; sütunlar Ocak-Aralık 1960): artı değer = model geride kalmış
round(rbind(ETS = a - as.numeric(fc_ets$mean),
            HW  = a - as.numeric(fc_hw$mean)), 1)
#> ETS  5.1 -16.0 -48.3 10.3 20.5 21.8 52.1  38.4  11.9  28.8  13.4   7.2
#> HW   0.4  -2.7 -43.3 12.5 -0.2 -5.0 -3.6 -29.4 -12.6   5.8  -9.7  -8.3

# Sönümlü trendin toplam katkısı (phi = 0.98): 12 ve 60 adımda, sonsuzda
phi <- 0.98
c(sum(phi^(1:12)), sum(phi^(1:60)), phi / (1 - phi))
#> [1] 10.54888 34.41990 49.00000

# Theta = SES + sürüklenme: aylık eğim ve SES katsayısı
fc_theta$model$drift
#> 1.276943
round(fc_theta$model$alpha, 3)
#> [1] 0.907

# Theta'yı log ölçekte kurmak (Bölüm 7.6'daki log dönüşümü)
round(metrikler(a, as.numeric(exp(thetaf(log(egitim), h = h)$mean))), 2)
#>   MAE  RMSE  MAPE
#> 14.94 21.48  3.24

# ---- Hemen deneyin 7.8.6: sönümlü ve sönümsüz trend ----
fit_nd <- ets(egitim, damped = FALSE)                 # sönümsüz trende zorla
print(fit_nd)
#> ETS(M,A,M)   alpha = 0.4415   beta = 4e-04   gamma = 0.4696
c(sonumlu = fit_ets$aicc, sonumsuz = fit_nd$aicc)
#>  sonumlu sonumsuz
#> 1250.511 1262.941
accuracy(forecast(fit_nd, h = h), test)[, c("MAE", "RMSE", "MAPE")]
#> Test set     10.808561 16.45869 2.345369

# Şimdi siz: log ölçekte ETS
fit_log <- ets(log(egitim))
print(fit_log)
#> ETS(A,A,A)   alpha = 0.7468   beta = 1e-04   gamma = 1e-04
round(metrikler(a, as.numeric(exp(forecast(fit_log, h = h)$mean))), 2)
#>   MAE  RMSE  MAPE
#> 16.11 23.49  3.57
