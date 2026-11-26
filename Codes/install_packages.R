# Ders notundaki R kodları için gerekli paketleri kurar.
# Kullanım (depo kök dizininde R/RStudio konsolunda):
#   source("Codes/install_packages.R")
# Yalnızca eksik olan paketler kurulur.

paketler <- c(
  "lubridate",  # Bölüm 4: tarih ve zaman işlemleri
  "xts",        # Bölüm 5: düzensiz aralıklı zaman serileri (zoo ile birlikte kurulur)
  "zoo",
  "TSstudio",   # Bölüm 5, 6: USgas veri seti ve ts_info()
  "ggplot2",    # Bölüm 6, 8: grafikler
  "tseries",    # Bölüm 3, 7: adf.test(), kpss.test()
  "forecast"    # Bölüm 7, 8: auto.arima(), forecast(), accuracy()
)

eksik <- paketler[!paketler %in% rownames(installed.packages())]
if (length(eksik) > 0) {
  install.packages(eksik)
} else {
  message("Tüm paketler zaten kurulu.")
}
