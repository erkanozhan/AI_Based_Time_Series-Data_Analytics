[İçindekiler](../Course_notes.md) · [← 16. TimeSeriesSplit: Zaman Serisinde Çapraz Doğrulama](16_timeseriessplit_zaman_serisinde_capraz_dogrulama.md)

## 17. Zaman Serisi Tahmininde 10 Altın Kural

Bu son bölüm, ders boyunca gördüğümüz yöntemlerden bağımsız olarak geçerli olan temel ilkeleri bir araya getirir. Algoritmalar (ARIMA, Prophet, XGBoost, LSTM vb.) değişse de bu kurallar değişmez. Her kuralın yanında, konunun ayrıntılı olarak işlendiği bölüme atıf verilmiştir.

---

### 17.1. Görsel İnceleme Tartışılamaz (Visual Inspection is Non-Negotiable)

Herhangi bir modelleme kodu yazmadan önce veriyi mutlaka grafiğe dökün. Özet istatistikler yanıltabilir, ama grafikler nadiren yanıltır. Grafikte şunları arayın:

- **Trend:** Veri yukarı mı, aşağı mı hareket ediyor?
- **Mevsimsellik:** Tekrarlayan bir desen var mı? Dalgaların genliği seviyeyle birlikte büyüyor mu (toplamsal mı, çarpımsal mı)?
- **Aykırı değerler (outliers):** Olmaması gereken ani sıçramalar var mı?
- **Boşluklar:** Eksik veri var mı?

Neden özet istatistik yetmez? İki seri düşünün: A serisi 1, 2, 3, 4, 5 ve B serisi 5, 4, 3, 2, 1. İkisinin de ortalaması $(1 + 2 + 3 + 4 + 5) / 5 = 3$'tür ve en küçük, en büyük değerleri de aynıdır. Ama biri sürekli artan, öteki sürekli azalan bir seridir; bunu ancak sırayı, yani grafiği görünce anlarsınız. Zaman serisinde bilginin önemli bir kısmı değerlerin **sırasındadır**.

Bileşenler ve ayrıştırma Bölüm 2'de, görselleştirme ve ACF/PACF grafikleri Bölüm 6'da ele alınmıştır.

### 17.2. Veriyi Asla Karıştırmayın (Never Shuffle Your Data)

Standart makine öğrenmesinde eğitim/test ayrımı için veriyi karıştırmak (shuffle) yaygındır, ancak zaman serilerinde bu büyük bir hatadır. Zaman tek yönde akar; bugünü tahmin etmek için gelecek haftanın verisini kullanamazsınız. Daima **zamansal ayrım (temporal split)** kullanın:

- *Örnek:* **Eğitim:** Ocak 2020 – Aralık 2023 | **Test:** Ocak 2024 – Mart 2024

Karıştırmanın ne yaptığını küçük bir örnekle görelim. Bir dükkânın 6 aylık satışları 10, 12, 14, 16, 18, 20 olsun. Veriyi karıştırıp 3. ayı (14) teste ayırırsak model 2. ay (12) ile 4. ayı (16) eğitimde görür ve 14'ü ikisinin ortasından kolayca "tahmin eder". Gerçek hayatta ise 4. ayı bildiğiniz bir anda 3. ayı tahmin etmeniz gerekmez; test her zaman bilinen son günden **sonrası** olmalıdır.

Eğitim-test ayrımı Bölüm 8'de, zaman sırasını koruyan çapraz doğrulama (TimeSeriesSplit) Bölüm 16'da anlatılmıştır.

> **Hemen deneyin 17.2 · Karıştırınca otokorelasyon kaybolur** (R, ~3 dk)
>
> Bölüm 1.2'deki kutuda karıştırınca komşu aylar arasındaki korelasyonun 0.96'dan 0.03'e düştüğünü gördük. Aynı deneyi ACF ve güven sınırıyla, yıllık gecikme (12) dahil yapın.
>
> ```r
> set.seed(1)
> karisik <- sample(as.numeric(AirPassengers))          # sıra rastgele bozuldu
> round(acf(AirPassengers, plot = FALSE)$acf[c(2, 13)], 2)   # gecikme 1 ve 12
> round(acf(karisik, plot = FALSE)$acf[c(2, 13)], 2)
> round(1.96 / sqrt(144), 2)                            # %95 güven sınırı (Bölüm 6.3.3)
> ```
>
> `acf(..., plot = FALSE)$acf` korelasyonları bir liste olarak verir; ilk eleman gecikme 0'dır (her zaman 1), bu yüzden 2. eleman gecikme 1'e, 13. eleman gecikme 12'ye karşılık gelir.
>
> **Ne göreceksiniz:**
>
> ```text
> [1] 0.95 0.76
> [1] 0.03 0.12
> [1] 0.16
> ```
>
> Gerçek seride geçen ayla korelasyon 0.95, geçen yılın aynı ayıyla 0.76'dır (Bölüm 6'daki değerler). Karıştırılınca ikisi de güven sınırının (±0.16) içine düşer: "geçen ay" ve "geçen yıl" artık anlamsızdır. Karıştırılmış veriyle eğitilen bir model bu yapıyı hiç göremez.
>
> **Şimdi siz:** `set.seed(2)`, `set.seed(3)` ve `set.seed(4)` ile tekrarlayın. Ayrıca `acf(karisik, lag.max = 24)` ile grafiği çizin: 24 gecikmeden kaçı bandın dışına taşar? Bu, "karıştırma yapıyı yok etti" sonucunu bozar mı?
>
> <details><summary>Cevap</summary>
>
> Gecikme 1 ve 12 değerleri tohumdan tohuma değişir ama hep küçüktür (bizde tohum 2: −0.05 ve 0.12; tohum 3: −0.15 ve −0.05; tohum 4: 0.10 ve 0.02). 24 gecikmede bandı aşan çubuk sayısı 0–2 arasındadır (tohum 1'de yalnızca gecikme 5). Bant %95'lik olduğu için tamamen rastgele bir seride 24 gecikmenin yaklaşık 1'inin şans eseri dışarı taşması beklenir; bu, yapının yok olduğu sonucunu değiştirmez.
>
> </details>

### 17.3. Bir Referans Noktası Belirleyin (Establish a Baseline: The Naive Model)

Karmaşık bir modelin (örneğin LSTM) gerçekten "iyi" olup olmadığını anlamak için bir kıyaslama noktasına ihtiyacınız vardır. Modelinizi daima **saf yöntem (naive method)** ile karşılaştırın:

- **Naive 1:** Yarının değeri, bugünün değeri ile aynı olacaktır.
- **Naive 2 (mevsimsel):** Önümüzdeki Haziran ayının satışları, geçen Haziran ile aynı olacaktır.
- *Kural:* Karmaşık modeliniz bu basit yöntemleri geçemiyorsa, canlıya almaya değmez.

*Sayısal örnek:* Son üç günün satışı 100, 120, 110 ise naive tahmin yarın için 110'dur; hiçbir hesap gerektirmez. Yarın gerçekleşen satış 115 olursa naive yöntemin hatası $115 - 110 = 5$ olur. Aylarca uğraşılan bir LSTM modeli aynı gün 108 tahmin etmişse hatası $115 - 108 = 7$'dir ve naive yöntemden kötüdür. Bir günlük karşılaştırma elbette yetmez, ama uzun bir test döneminde de durum böyleyse karmaşık modelin bir katkısı yoktur.

Referans modelle karşılaştırma ve hata metrikleri Bölüm 8.1.2'de ele alınmıştır.

> **Hemen deneyin 17.3 · Borsa endeksinde naive'i yenmeye çalışın** (R, ~4 dk)
>
> DAX endeksinin 1991–1998 günlük kapanışlarında `auto.arima()` ile seçilen modeli, son 300 iş gününde "yarın = bugün" kuralıyla karşılaştırın. Her iki yöntem de her gün bir önceki günün gerçek değerini görür (tek adımlı tahmin).
>
> ```r
> library(forecast)
> dax <- ts(as.numeric(EuStockMarkets[, "DAX"]))   # 1860 iş günlük DAX endeksi (1991-1998)
> fit <- auto.arima(window(dax, end = 1560))        # model ilk 1560 günle seçilir
> fit
> tum <- Arima(dax, model = fit)                    # aynı katsayılarla tüm seride 1 adım ileri tahmin
> test <- 1561:1860                                 # son 300 iş günü
> hata_arima <- dax[test] - fitted(tum)[test]
> hata_naive <- dax[test] - dax[test - 1]           # naive: yarın = bugün
> round(c(ARIMA = sqrt(mean(hata_arima^2)), Naive = sqrt(mean(hata_naive^2))), 1)
> ```
>
> `Arima(dax, model = fit)` modeli yeniden tahmin etmez; ilk 1560 günde bulunan katsayıları serinin tamamına uygular. `fitted()` her gün için bir önceki güne kadarki bilgiyle yapılan tahmini verir.
>
> **Ne göreceksiniz:** `auto.arima` `ARIMA(2,2,0)` seçer (iki kez fark alınmış, iki AR terimli bir model). Test sonucu:
>
> ```text
> ARIMA Naive 
>  77.8  67.1 
> ```
>
> Hiç parametresi olmayan naive yöntem, otomatik seçilmiş ARIMA'dan yaklaşık %14 daha düşük hata verir. Finansal fiyat serilerinde bu çok yaygın bir durumdur: dünkü fiyat, yarının en iyi tahminlerinden biridir.
>
> **Şimdi siz:** `auto.arima(window(dax, end = 1560), d = 1)` yazarak fark derecesini 1'e sabitleyin. Hangi model seçilir ve naive'e göre durumu ne olur?
>
> <details><summary>Cevap</summary>
>
> `ARIMA(0,1,0) with drift` seçilir: Bu, "yarın = bugün + küçük sabit artış (1.39 puan)" demektir, yani naive'in kayma terimli hâlidir. RMSE'ler `67.0` ve `67.1`, pratikte aynı. En iyi ARIMA, naive'e eşit oldu; onu yenemedi. Karmaşık bir model kurmadan önce referansı ölçmek bu yüzden şarttır.
>
> </details>

### 17.4. Durağanlığa Saygı Gösterin (Respect Stationarity)

Çoğu klasik istatistiksel model (ARIMA, VAR gibi), serinin istatistiksel özelliklerinin (ortalama, varyans) zaman içinde değişmemesini varsayar. **Ortalama** serinin tipik düzeyidir; **varyans** ise değerlerin bu düzeyin çevresinde ne kadar geniş dalgalandığının ölçüsüdür. Ortalaması ve dalgalanma genişliği zamanla değişmeyen seriye **durağan** denir.

- Veride trend varsa farkını alın (difference it): Her değerden bir öncekini çıkarın. 100, 110, 125, 135 serisinin farkları $110 - 100 = 10$, $125 - 110 = 15$, $135 - 125 = 10$ olur; sürekli yükselen seri, 10–15 civarında dolaşan bir seriye dönüşür.
- Mevsimsellik varsa mevsimsel fark alın: Aylık veride her değerden bir yıl önceki aynı ayın değerini çıkarın (ör. Temmuz 1950 eksi Temmuz 1949: $170 - 148 = 22$).
- Varyans seviyeyle birlikte artıyorsa logaritmik dönüşüm uygulayın. Logaritma büyük sayıları küçüklere göre daha çok sıkıştırır; böylece seri büyüdükçe genişleyen dalgalar eşitlenir. Örneğin 10 tabanında 100'ün logaritması 2, 1000'inki 3'tür: 900'lük fark 1'lik farka iner.
- Durağanlığı yalnızca gözle değil, ADF ve KPSS gibi testlerle de doğrulayın.

Durağanlığın tanımı Bölüm 2.5 ve 3.2'de, ADF ve KPSS testleri Bölüm 7.5'te, fark alma ve log dönüşümünün ARIMA/SARIMA'da uygulanması Bölüm 7.6'da işlenmiştir. Makine öğrenmesi modelleri durağanlık varsaymasa da, trendli serilerde fark almanın ağaç tabanlı modellere nasıl yardımcı olduğu Bölüm 12.1.4 ve 13.4.2'de tartışılmıştır.

> **Hemen deneyin 17.4 · Log dönüşümü dalgaları eşitler mi?** (R, ~4 dk)
>
> AirPassengers'ın aylık değişimlerinin oynaklığını ilk iki yıl ile son iki yıl arasında karşılaştırın; önce ham seride, sonra log seride. Ardından kaç fark gerektiğini sorun.
>
> ```r
> library(forecast); library(tseries)
> x <- AirPassengers
> fark <- diff(x)                                   # ham serinin aylık değişimleri
> round(c(sd(window(fark, end = c(1950, 12))), sd(window(fark, start = c(1959, 1)))), 1)
> lfark <- diff(log(x))                             # log serinin aylık değişimleri
> round(c(sd(window(lfark, end = c(1950, 12))), sd(window(lfark, start = c(1959, 1)))), 3)
> ndiffs(log(x)); nsdiffs(log(x))                   # kaç normal, kaç mevsimsel fark gerekir?
> kpss.test(diff(diff(log(x), lag = 12)))$p.value   # H0: seri durağandır
> ```
>
> **Ne göreceksiniz:**
>
> ```text
> [1] 14.8 52.7
> [1] 0.112 0.112
> [1] 1
> [1] 1
> [1] 0.1
> ```
>
> Ham seride aylık değişimlerin standart sapması 1949–1950'de 14.8, 1959–1960'ta 52.7'dir: dalgalar seviyeyle birlikte yaklaşık 3.6 kat büyümüştür. Log seride iki dönem neredeyse aynıdır (üç basamakta 0.112; tam değerler 0.1120 ve 0.1123): log dönüşümü varyansı sabitlemiştir. `ndiffs` ve `nsdiffs` birer normal ve mevsimsel fark önerir; iki fark alınmış log serinin KPSS p-değeri 0.1'dir (durağanlık reddedilmez). Altta çıkan "p-value greater than printed p-value" uyarısı, gerçek p-değerinin tablonun en büyük değeri olan 0.1'den de büyük olduğunu söyler.
>
> **Şimdi siz:** Fark almadan `kpss.test(log(x))$p.value` hesaplayın. Sonucu ve uyarıyı yorumlayın.
>
> <details><summary>Cevap</summary>
>
> `0.01` ve "p-value smaller than printed p-value" uyarısı: Gerçek p-değeri 0.01'den de küçüktür, durağanlık hipotezi açıkça reddedilir. Log dönüşümü varyansı eşitledi ama trendi ve mevsimselliği gidermedi; bunun için farklar gerekir (Bölüm 7.6'daki $d = 1$, $D = 1$ seçimi).
>
> </details>

### 17.5. Alan Bilgisi > Algoritmalar (Domain Knowledge > Algorithms)

Bir algoritma, satışlardaki ani artışın "Kara Cuma" (Black Friday) yüzünden olduğunu veya düşüşün bir sunucu kesintisinden kaynaklandığını kendi başına bilemez.

- **Özellik mühendisliği:** Tatilleri, hava durumunu veya pazarlama etkinliklerini dışsal değişkenler olarak modele ekleyin. Bağlam (context), genellikle hiperparametre optimizasyonundan daha güçlüdür.

Prophet'ta tatil etkileri Bölüm 9'da, gecikme ve takvim özellikleriyle özellik mühendisliği Bölüm 12 ve 13'te ele alınmıştır.

> **Hemen deneyin 17.5 · Emniyet kemeri yasası: alan bilgisini modele koyun** (R, ~4 dk)
>
> R'daki `Seatbelts` verisi İngiltere'de 1969–1984 arası aylık trafik kazası kayıplarını içerir. 31 Ocak 1983'te çıkan yasayla ön koltukta emniyet kemeri takmak zorunlu oldu. Algoritma bunu bilemez; biz söyleyebiliriz.
>
> ```r
> library(forecast)
> y   <- Seatbelts[, "front"]     # İngiltere: ön koltukta ölen/ağır yaralanan yolcu (aylık, 1969-1984)
> law <- Seatbelts[, "law"]       # 0: kemer zorunlu değil, 1: zorunlu (Şubat 1983'ten itibaren)
> m1 <- tslm(y ~ trend + season)          # algoritma: yalnızca trend + mevsim
> m2 <- tslm(y ~ trend + season + law)    # + alan bilgisi: yasa ne zaman çıktı?
> round(c(CV(m1)["AdjR2"], CV(m2)["AdjR2"]), 3)
> round(coef(m2)["law"], 1)
> round(mean(window(residuals(m1), start = c(1983, 2))), 1)   # yasa sonrası ortalama artık
> ```
>
> `tslm()` trend ve aylık kukla değişkenlerle doğrusal regresyon kurar; `CV()` modelin uyum ölçütlerini, `AdjR2` düzeltilmiş $R^2$'yi verir.
>
> **Ne göreceksiniz:**
>
> ```text
> AdjR2 AdjR2 
> 0.705 0.753 
>    law 
> -141.4 
> [1] -84.8
> ```
>
> Tek bir 0/1 sütunu açıklama gücünü 0.705'ten 0.753'e çıkarır. Model, yasanın ayda yaklaşık 141 ön koltuk kaybını azalttığını tahmin eder. Yasayı bilmeyen `m1` ise yasa sonrasındaki 23 ayda her ay ortalama 85 kişi **fazla** tahmin eder (artık ortalaması −84.8): sistematik bir hata.
>
> **Şimdi siz:** Aynı ortalamayı `m2`'nin artıkları için hesaplayın. Ne çıkmasını beklersiniz?
>
> <details><summary>Cevap</summary>
>
> `round(mean(window(residuals(m2), start = c(1983, 2))), 1)` → `0`. Yasa değişkeni modele girince yasa sonrası dönemdeki sistematik sapma tamamen ortadan kalkar: Hiperparametre ayarlamak bunu yapamazdı, alan bilgisi yaptı.
>
> </details>

### 17.6. Veri Sızıntısına Dikkat Edin (Watch Out for Leakage)

Zaman serilerinde veri sızıntısı sinsi olabilir. Model eğitilirken tahmin anında bilinemeyecek bir gelecek bilgisi kullanılırsa, model eğitimde harika görünür ama üretimde (production) çuvallar.

- *Örnek:* Ocak 2024'ün günlük satışlarını tahmin etmek için Ocak 2024'ün "aylık ortalama sıcaklığını" kullanmak. (Ay bitene kadar aylık ortalamayı bilemezsiniz!)
- *Diğer sık hatalar:* Ölçekleyiciyi tüm veriyle uydurmak (Bölüm 15.1.1 ve 16.4), hareketli ortalamayı `shift(1)` olmadan hesaplamak, sıradan k-katlı çapraz doğrulama kullanmak (Bölüm 16.1).

`shift(1)` hatasının sayısal örneği Bölüm 12.1.5'tedir: `shift(1)` olmadan hesaplanan hareketli ortalama, tahmin etmeye çalıştığımız değeri zaten içerir. Model eğitimde harika görünür, çünkü cevabın bir parçası sorunun içindedir; ama yarın için bu özelliği hesaplayamazsınız, çünkü yarının satışını henüz bilmiyorsunuz.

Zamansal ayrım Bölüm 8'de, sızıntı türleri Bölüm 12.1.5'te, zaman serisine uygun çapraz doğrulama Bölüm 16'da anlatılmıştır.

### 17.7. Diyagnostikler Önemlidir: Hataları Kontrol Edin (Diagnostics Matter)

İyi bir model tüm "sinyali" alır ve geriye yalnızca "gürültü" bırakır. Modelin artıklarını (hatalarını) kontrol edin. **Artık**, her dönem için gerçek değer eksi tahmindir. Hatalar **beyaz gürültü (white noise)** gibi, yani önceden kestirilemeyen, birbirinden bağımsız küçük sapmalar gibi görünmelidir:

- Ortalama sıfır olmalı.
- Varyans sabit olmalı.
- Otokorelasyon olmamalı (hataların ACF grafiğine bakın).
- *Hatalarda bir desen varsa, modeliniz bir şeyi gözden kaçırmış demektir.*

*Örnek:* Altı ayın artıkları +5, +7, +6, +8, +6, +7 ise hepsi pozitiftir ve ortalamaları $(5 + 7 + 6 + 8 + 6 + 7) / 6 = 39 / 6 = 6.5$'tir. Model her ay yaklaşık 6.5 birim düşük tahmin ediyordur; bu bir desendir ve düzeltilebilir (ör. trend eksik modellenmiş olabilir). Artıklar +3, −2, +1, −4, +2, 0 gibi işaret değiştirerek sıfır çevresinde dağılıyorsa (ortalama $0 / 6 = 0$) modelin kaçırdığı belirgin bir yapı görünmez. Bölüm 15.4.2'deki CNN'in "Ortalama hata" satırı bu kontrolün bir örneğidir.

ACF grafiğinin okunması Bölüm 6'da, ARIMA artıklarının kontrolü Bölüm 7'de gösterilmiştir.

> **Hemen deneyin 17.7 · Ljung-Box, eksik mevsimselliği yakalar** (R, ~4 dk)
>
> Log AirPassengers'a mevsimsel kısmı olmayan bir ARIMA ve Bölüm 7.6'daki havayolu modelini kurun, artıkları karşılaştırın.
>
> ```r
> library(forecast)
> y  <- log(AirPassengers)
> m1 <- Arima(y, order = c(0, 1, 1))                          # mevsimsel kısım YOK
> m2 <- Arima(y, order = c(0, 1, 1), seasonal = c(0, 1, 1))   # havayolu modeli (Bölüm 7.6)
> checkresiduals(m1, plot = FALSE)
> checkresiduals(m2, plot = FALSE)
> round(acf(residuals(m1), plot = FALSE)$acf[13], 2)          # artıkların lag-12 korelasyonu
> round(acf(residuals(m2), plot = FALSE)$acf[13], 2)
> ```
>
> **Ne göreceksiniz** (kısaltılmış):
>
> ```text
> data:  Residuals from ARIMA(0,1,1)
> Q* = 261.81, df = 23, p-value < 2.2e-16
>
> data:  Residuals from ARIMA(0,1,1)(0,1,1)[12]
> Q* = 26.446, df = 22, p-value = 0.233
>
> [1] 0.81
> [1] -0.05
> ```
>
> İlk modelin artıklarında 12 ay önceyle korelasyon 0.81'dir: Mevsimsellik hatalara kaçmıştır ve Ljung-Box bunu p < 2.2e-16 ile yakalar. Havayolu modelinin artıklarında bu korelasyon −0.05'e iner ve p = 0.233 çıkar (Bölüm 7.6.6'daki değer): Artıklar beyaz gürültüden ayırt edilemez. `plot = FALSE`'u silerseniz ilk modelin artık ACF'sinde 12, 24, … gecikmelerinde uzun çubuklar görürsünüz.
>
> **Şimdi siz:** Mevsimsel farkı alıp mevsimsel MA terimini koymayan `seasonal = c(0, 1, 0)` modelini deneyin. Ljung-Box geçer mi? Artıkların lag-12 korelasyonu ne söyler?
>
> <details><summary>Cevap</summary>
>
> Geçmez: `Q* = 59.533, df = 23, p-value = 4.462e-05`. Lag-12 korelasyonu −0.42'dir: Mevsimsel fark fazla düzeltme yapmış, artıklarda negatif bir yıllık ilişki bırakmıştır. Bu desen, tam da $\Theta_1$ (mevsimsel MA) teriminin giderdiği yapıdır.
>
> </details>

### 17.8. Belirsizliği Kucaklayın (Embrace Uncertainty)

Nokta tahminleri (ör. "Satışlar 105 adet olacak") neredeyse her zaman bir miktar yanlıştır. Bunun yerine karar vericilerin riski değerlendirebilmesi için **tahmin aralıkları (prediction intervals)** sunun:

- *Örnek:* "Satışlar %95 olasılıkla 95 ile 115 adet arasında olacak."

Bu cümlenin pratik anlamı şudur: Model bu biçimde 100 tahmin aralığı verirse, gerçekleşen değerin bunların yaklaşık 95'inde aralığın içine düşmesi beklenir. Aralığın genişliği belirsizliğin ölçüsüdür: "95–115" ile "60–150" aynı nokta tahminine (105) sahip olabilir, ama ikincisi karar verici için çok daha riskli bir duruma işaret eder. Depo planlayan biri, ilk durumda 115 adetlik stokla rahat ederken ikinci durumda çok daha temkinli davranmalıdır.

Aralığın düzeyini de mutlaka belirtin: Örneğin Prophet'ın `yhat_lower`/`yhat_upper` sütunları varsayılan olarak %80'lik aralığı verir (Bölüm 9). ARIMA'nın tahmin aralıkları Bölüm 7'de ele alınmıştır.

> **Hemen deneyin 17.8 · Tahmin aralığı ufukla genişler** (R, ~3 dk)
>
> Nil Nehri'nin yıllık akışı için naive tahmin üretin ve aralıkların 1, 4 ve 16 yıl ileride nasıl genişlediğine bakın.
>
> ```r
> library(forecast)
> f <- naive(Nile, h = 16, level = c(80, 95))   # yarın = bugün; %80 ve %95 aralık
> tablo <- round(as.data.frame(f))              # tahmin tablosu: her satır bir gelecek yıl
> tablo[c(1, 4, 16), ]                          # 1, 4 ve 16 yıl ileri
> tablo$`Hi 95`[c(1, 4, 16)] - tablo$`Lo 95`[c(1, 4, 16)]   # %95 aralığın genişliği
> ```
>
> **Ne göreceksiniz:**
>
> ```text
>      Point Forecast Lo 80 Hi 80 Lo 95 Hi 95
> 1971            740   526   954   412  1068
> 1974            740   311  1169    84  1396
> 1986            740  -118  1598  -572  2052
> [1]  656 1312 2624
> ```
>
> Nokta tahmini hep 740'tır (1970'in değeri), ama belirsizlik hızla büyür. Genişlik 4 yıl ileride iki katına, 16 yıl ileride dört katına çıkar: Naive yöntemde aralık $\sqrt{h}$ ile genişler ($\sqrt{4} = 2$, $\sqrt{16} = 4$). %80'lik aralık her zaman %95'likten dardır. 1986 satırındaki eksi alt sınırlar da bir uyarıdır: Akış negatif olamaz; model bu fiziksel sınırı bilmez.
>
> **Şimdi siz:** 9 yıl ilerideki %95'lik aralığın genişliğini çalıştırmadan tahmin edin, sonra `tablo[9, ]` ile kontrol edin. %80'lik aralığın 1. yıldaki genişliği nedir?
>
> <details><summary>Cevap</summary>
>
> $\sqrt{9} = 3$ olduğundan yaklaşık $3 \cdot 656 = 1968$. `tablo[9, ]` 1979 satırını verir: `Lo 95` = −244, `Hi 95` = 1724, genişlik $1724 - (-244) = 1968$. %80'lik aralık 1. yılda yaklaşık 429 genişliğindedir (tabloda $954 - 526 = 428$, yuvarlama farkı); 4. yılda 858, 16. yılda 1715. Aynı $\sqrt{h}$ kuralı, daha dar bir aralıkla.
>
> </details>

### 17.9. Doğru Metriği Seçin (Choose the Right Metric)

Yalnızca R² değerine güvenmeyin. **R²** ("R kare"), modelin serideki değişkenliğin ne kadarını açıkladığını 0 ile 1 arasında bir sayıyla özetler; 1'e yakın olması iyi sayılır. Trendli serilerde ise bu ölçüt kolayca yanıltır: AirPassengers'ta hiçbir şey öğrenmeyen naive yöntem bile ("bu ay = geçen ay") yaklaşık 0.92'lik bir R² verir, çünkü seviyesi sürekli artan bir seride dünkü değer bugünün düzeyini zaten büyük ölçüde belirler. İş durumunuza uygun metriği seçin:

- **RMSE:** Büyük hataları ağır cezalandırır (büyük sapmaların kritik olduğu tahminler için iyidir).
- **MAE:** Yorumlaması daha kolaydır (hataların ortalama büyüklüğü, serinin kendi biriminde).
- **MAPE:** Yüzdelik olduğu için farklı ölçekteki serileri karşılaştırmaya uygundur, ancak gerçek değerler sıfır ya da sıfıra çok yakınsa kullanılamaz.

*Sayısal örnek:* Beş günün mutlak hataları 1, 1, 1, 1 ve 10 ise MAE $= 14 / 5 = 2.8$, RMSE $= \sqrt{104 / 5} \approx 4.56$ olur: Tek bir büyük hata RMSE'yi MAE'nin 1.6 katına çıkarır. Büyük hatalar sizin için özellikle pahalıysa (ör. hastane yatak planlaması) RMSE'ye bakmak bu yüzden anlamlıdır. MAPE'nin sıfıra yakın değerlerde nasıl patladığı Bölüm 8.2.3'te sayılarla gösterilmiştir.

Bu metriklerin formülleri, karşılaştırması ve Python uygulaması Bölüm 8.2'de verilmiştir.

> **Hemen deneyin 17.9 · Hiçbir şey öğrenmeyen modelin R²'si** (R, ~3 dk)
>
> Metindeki iddiayı doğrulayın: AirPassengers'ta "bu ay = geçen ay" tahmininin $R^2$'si yaklaşık 0.92 mi?
>
> ```r
> x <- as.numeric(AirPassengers)
> gercek <- x[-1]                    # Şubat 1949 - Aralık 1960
> naive  <- x[-length(x)]            # tahmin = bir önceki ayın değeri (hiçbir şey öğrenmeyen model)
> R2 <- 1 - sum((gercek - naive)^2) / sum((gercek - mean(gercek))^2)
> round(R2, 2)
> round(c(MAE = mean(abs(gercek - naive)), RMSE = sqrt(mean((gercek - naive)^2)),
>         MAPE = 100 * mean(abs(gercek - naive) / gercek)), 2)
> ```
>
> `x[-1]` ilk elemanı, `x[-length(x)]` son elemanı atar; böylece her gerçek değerin yanına bir önceki ay gelir. $R^2$, hataların karelerinin toplamını serinin kendi ortalaması çevresindeki dalgalanmayla kıyaslar.
>
> **Ne göreceksiniz:**
>
> ```text
> [1] 0.92
>   MAE  RMSE  MAPE 
> 25.86 33.71  9.02 
> ```
>
> $R^2 = 0.92$ "harika" görünür, oysa model hiçbir şey öğrenmemiştir. Ortalama mutlak hata ise 26 bin yolcu, yüzde hata yaklaşık %9'dur: İş açısından anlamlı olan bu sayılardır.
>
> **Şimdi siz:** Mevsimsel naive'i ("bu ay = geçen yılın aynı ayı") deneyin: `gercek <- x[-(1:12)]` ve `naive <- x[1:132]` yazıp son üç satırı yeniden çalıştırın. Hangi metrikler daha iyi, hangileri daha kötü çıkar?
>
> <details><summary>Cevap</summary>
>
> $R^2$ = 0.90; MAE = 32.03, RMSE = 36.32, MAPE = 11.25. Tüm sıradan metriklere göre mevsimsel naive bu seride (bütün dönemde) basit naive'den biraz kötüdür, çünkü bir yıllık trend artışını hiç yakalamaz. İki $R^2$ ise birbirine çok yakın (0.92 ve 0.90): $R^2$ trendli seride modelleri ayırt etmekte zayıftır.
>
> </details>

### 17.10. Karmaşıklık ≠ Doğruluk (Complexity ≠ Accuracy)

Her problem için en yeni Transformer veya derin öğrenme modelini kullanma eğilimi vardır. Oysa birçok gerçek dünya tek değişkenli (univariate) zaman serisinde üstel düzeltme (ETS) veya ARIMA gibi basit modeller, karmaşık sinir ağlarından daha iyi performans gösterir.

- Basit başlayın; karmaşıklığı ancak basit modeller yetersiz kaldığında artırın.

Bu dersin kendi sonuçları da bunu gösterir: Bölüm 15.5'teki karşılaştırmada yalnızca birkaç parametresi olan ARIMA modeli, 60 ay ileriye tahmin gibi çok daha zor bir görevde bile, her ay gerçek geçmişi gören 10 binden fazla parametreli LSTM ile aynı düzeyde hata verdi. Aynı GRU mimarisi tohum değiştiğinde 47 ile 184 arasında değişen hatalar üretti; 1D-CNN'de iki evrişim katmanının arkasına birer `BatchNormalization` katmanı eklemek ise test hatasını 32'den 248'e çıkardı. Karmaşık model daha fazla veri, daha fazla ayar ve daha fazla denetim ister; bunlar yoksa üstünlüğü kâğıt üzerinde kalır.

Klasik, makine öğrenmesi ve derin öğrenme yaklaşımlarının karşılaştırması Bölüm 12.3'te, derin öğrenme modellerinin karşılaştırmalı uygulaması Bölüm 15'te yer almaktadır.

---

### 17.11. Kapanış

Bu on kural, ders boyunca izlediğimiz yolun özetidir: Veriyi önce **görün** ve anlayın (Bölüm 1–6), basit ve yorumlanabilir modellerle **başlayın** (Bölüm 7–11), her modeli **dürüst** bir test düzeniyle ve bir referans modele karşı **ölçün** (Bölüm 8 ve 16), karmaşık yapay zeka modellerine ise ancak gerçekten katkı sağladıklarında **geçin** (Bölüm 12–15). Hangi algoritmayı kullanırsanız kullanın, iyi bir tahminin sırrı çoğu zaman modelden çok veriyi, zamanın yönünü ve belirsizliği doğru ele almaktadır.

**Kaynak:** https://ozancanozdemir.github.io/posts/2025/12/10-rules-time-series-forecasting/

---

[İçindekiler](../Course_notes.md) · [← 16. TimeSeriesSplit: Zaman Serisinde Çapraz Doğrulama](16_timeseriessplit_zaman_serisinde_capraz_dogrulama.md)
