# Yapay Zeka Tabanlı Zaman Serisi ve Veri Analizi

### Ders Notları

## İçindekiler

1. [Zaman Serisi Analizine Giriş](notlar/01_zaman_serisi_analizine_giris.md)
2. [Zaman Serisinin Temel Kavramları ve Bileşenleri](notlar/02_zaman_serisinin_temel_kavramlari_ve_bilesenleri.md)
3. [Zaman Serisi Tipleri](notlar/03_zaman_serisi_tipleri.md)
4. [R'da Tarih ve Zaman Nesneleri](notlar/04_rda_tarih_ve_zaman_nesneleri.md)
5. [R'da Zaman Serisi Nesneleri: `ts` ve `xts`](notlar/05_rda_zaman_serisi_nesneleri_ts_ve_xts.md)
6. [Veri Manipülasyonu, Görselleştirme ve ACF/PACF](notlar/06_veri_manipulasyonu_gorsellestirme_ve_acf_pacf.md)
7. [Klasik İstatistiksel Modeller: ARIMA ve SARIMA](notlar/07_klasik_istatistiksel_modeller_arima_ve_sarima.md)
8. [Model Değerlendirme: Hata Metrikleri ve Eğitim-Test Ayrımı](notlar/08_model_degerlendirme_hata_metrikleri_ve_egitim_test_ayrimi.md)
9. [Facebook Prophet](notlar/09_facebook_prophet.md)
10. [VAR: Çok Değişkenli Zaman Serisi Modeli](notlar/10_var_cok_degiskenli_zaman_serisi_modeli.md)
11. [Gretl: Ekonometrik Analiz için Görsel Ortam](notlar/11_gretl_ekonometrik_analiz_icin_gorsel_ortam.md)
12. [Yapay Zeka ile Zaman Serisi Analizine Giriş](notlar/12_yapay_zeka_ile_zaman_serisi_analizine_giris.md)
13. [XGBoost ile Zaman Serisi Tahmini](notlar/13_xgboost_ile_zaman_serisi_tahmini.md)
14. [Weka Zaman Serisi Tahmin Modülü (Forecast Sekmesi)](notlar/14_weka_zaman_serisi_tahmin_modulu_forecast_sekmesi.md)
15. [Derin Öğrenme ile Tahmin: LSTM, GRU ve 1D-CNN](notlar/15_derin_ogrenme_ile_tahmin_lstm_gru_ve_1d_cnn.md)
16. [TimeSeriesSplit: Zaman Serisinde Çapraz Doğrulama](notlar/16_timeseriessplit_zaman_serisinde_capraz_dogrulama.md)
17. [Zaman Serisi Tahmininde 10 Altın Kural](notlar/17_zaman_serisi_tahmininde_10_altin_kural.md)

### Ekler

- [Ek A. R'a Giriş](R_Giris.md): R'ın temel yapıları, veri okuma ve temel grafikler (Bölüm 4'ten önce okunması önerilir)
- [Ek B. Gretl'e Giriş](Gretl.md): Gretl menüleriyle SARIMA, öngörü ve artıklara dayalı anomali tespiti (Bölüm 11'e hazırlık)

> **Kodlar ve veri:** Kısa kodlar metnin içindedir. Her bölümün kodlarının tamamı, çalıştırılabilir dosyalar olarak [`Codes/`](Codes/README.md) klasöründedir: R betikleri, Python betikleri ve kurulum gerektirmeden Colab'da açılabilen notebook'lar. Nasıl çalıştırılacakları [`Codes/README.md`](Codes/README.md) dosyasında anlatılır. Veri setleri [`data/`](data) klasöründedir.
>
> **Hemen deneyin kutuları:** Konuların içinde, anlatılanın okuyucunun kendi bilgisayarında hemen denenebilmesi için "Hemen deneyin" başlıklı kısa uygulamalar vardır. Her kutu birkaç dakikalık bir iştir ve hangi programla (R, Python, Gretl, Weka, Excel / LibreOffice Calc) yapılacağı başlığında yazar. Kutudaki kod ya da adımlar uygulanır, sonuç **Ne göreceksiniz** kısmıyla karşılaştırılır, ardından **Şimdi siz** görevi önce tahmin edilip sonra denenir; cevap, kutunun altındaki açılır bölümdedir. Kodlar depo kök dizininden çalıştırılacak şekilde yazılmıştır ve gerekli paketler [`Codes/README.md`](Codes/README.md)'de anlatıldığı gibi kurulur.

> Her bölüm, GitHub'da formüllerin eksiksiz görünmesi için ayrı bir dosyadadır (`notlar/` klasörü).

> **Yazım kuralları:** Ondalık ayırıcı olarak nokta kullanılır (R ve Python çıktılarıyla uyum için); Türkçe arayüzlü Excel ya da LibreOffice Calc çıktıları anlatılırken ekranda görünen virgüllü yazım korunur. Terimlerin İngilizce karşılıkları ilk geçtikleri yerde parantez içinde verilir. Gözlenen değer Bölüm 1–7'de $x_t$, model değerlendirme bölümlerinde (Bölüm 8 ve sonrası) makine öğrenmesi kaynaklarındaki alışkanlığa uyularak $y_t$ ile gösterilir; iki gösterim aynı anlamdadır.
