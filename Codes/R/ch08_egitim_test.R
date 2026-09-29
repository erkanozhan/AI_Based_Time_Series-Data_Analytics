# =============================================================================
# Bölüm 8 — Model Değerlendirme: Hata Metrikleri ve Eğitim-Test Ayrımı
# Ders notu: Course_notes.md, Bölüm 8.3 (R ile Eğitim-Test Uygulaması: AirPassengers)
#
# Çalıştırma: depo kök dizininden  Rscript Codes/R/ch08_egitim_test.R
#             ya da RStudio'da dosyayı açıp satır satır (Ctrl+Enter) çalıştırın.
# Gerekli paketler: forecast, ggplot2
#   (Kurulum için bir kez: source("Codes/install_packages.R"))
#
# Not: "#>" ile başlayan yorumlar, notta verilen beklenen çıktılardır.
#      Son adımdaki ggsave(), grafiği çalışma dizinine
#      arima_forecast_original_scale.png adıyla kaydeder.
# =============================================================================

library(forecast)
library(ggplot2)

data(AirPassengers)

# ---- 8.3 Eğitim ve test setleri (log ölçekte) ----
train <- window(log(AirPassengers), end = c(1959, 12))   # 1949-01 ... 1959-12 (132 ay)
test  <- window(log(AirPassengers), start = c(1960, 1))  # 1960-01 ... 1960-12 (12 ay)

# ---- 8.3 Modeli yalnızca eğitim setiyle kurma ----
# Model 1960'ı hiç görmüyor.
fit_train <- auto.arima(train, seasonal = TRUE)
print(fit_train)
#> ARIMA(0,1,1)(0,1,1)[12]
#>           ma1     sma1
#>       -0.3484  -0.5623

# ---- 8.3 Test dönemi için tahmin ve orijinal ölçeğe dönüş ----
# Test dönemi kadar (12 ay) ileriye tahmin
fc_test <- forecast(fit_train, h = length(test))

# Log ölçekten orijinal ölçeğe dönüş: logaritmanın tersi exp()
actual    <- as.numeric(exp(test))
predicted <- as.numeric(exp(fc_test$mean))

comparison <- data.frame(Ay = month.abb, Gercek = actual,
                         Tahmin = round(predicted, 1),
                         Hata = round(actual - predicted, 1))
print(comparison)

# ---- 8.3 Hata metrikleri (MAE, RMSE, MAPE) ----
# 8.2'deki formüllerin birebir karşılığı
mae  <- mean(abs(actual - predicted))
rmse <- sqrt(mean((actual - predicted)^2))
mape <- mean(abs((actual - predicted) / actual)) * 100
cat(sprintf("MAE = %.2f   RMSE = %.2f   MAPE = %%%.2f\n", mae, rmse, mape))
#> MAE = 13.26   RMSE = 18.59   MAPE = %2.90

# ---- 8.3 Naive referanslarla karşılaştırma ----
# Metrikleri tek satırda hesaplayan küçük bir yardımcı fonksiyon
metrikler <- function(a, p) c(MAE  = mean(abs(a - p)),
                              RMSE = sqrt(mean((a - p)^2)),
                              MAPE = mean(abs((a - p) / a)) * 100)

naive_fc  <- as.numeric(exp(naive(train,  h = 12)$mean))  # her ay = Aralık 1959
snaive_fc <- as.numeric(exp(snaive(train, h = 12)$mean))  # her ay = 1959'un aynı ayı

print(round(rbind(SARIMA            = metrikler(actual, predicted),
                  Naive             = metrikler(actual, naive_fc),
                  `Mevsimsel Naive` = metrikler(actual, snaive_fc)), 2))
#>                   MAE   RMSE  MAPE
#> SARIMA          13.26  18.59  2.90
#> Naive           76.00 102.98 14.25
#> Mevsimsel Naive 47.83  50.71  9.99

# forecast paketinin hazır fonksiyonu (DİKKAT: sonuçlar log ölçekte)
print(accuracy(fc_test, test))

# ---- 8.3 Görselleştirme: gerçek değerler ve tahminler (orijinal ölçek) ----
# Tarih sütunları oluştur (ggplot2 Date nesnesiyle daha iyi çalışır)
tum_tarihler  <- seq(as.Date("1949-01-01"), by = "month", length.out = length(AirPassengers))
test_tarihler <- seq(as.Date("1960-01-01"), by = "month", length.out = length(test))

plot_data <- rbind(
  data.frame(Tarih = tum_tarihler,  Deger = as.numeric(AirPassengers), Tur = "Gerçek (tüm seri)"),
  data.frame(Tarih = test_tarihler, Deger = actual,                    Tur = "Gerçek (test)"),
  data.frame(Tarih = test_tarihler, Deger = predicted,                 Tur = "SARIMA tahmini")
)

# (Rscript ile çalıştırırken ggplot nesnesinin çizilmesi için print() kullanıyoruz)
p <- ggplot(plot_data, aes(x = Tarih, y = Deger, color = Tur)) +
  geom_line() +
  labs(title = "AirPassengers: Gerçek Değerler ve Tahminler (Orijinal Ölçek)",
       y = "Yolcu Sayısı", x = "Yıl", color = NULL) +
  theme_minimal() +
  scale_color_manual(values = c("Gerçek (tüm seri)" = "black",
                                "Gerçek (test)"     = "red",
                                "SARIMA tahmini"    = "blue"))
print(p)

ggsave("arima_forecast_original_scale.png", plot = p, width = 10, height = 6, dpi = 300)
