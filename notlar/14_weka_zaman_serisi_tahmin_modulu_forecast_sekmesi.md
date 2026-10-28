[İçindekiler](../Course_notes.md) · [← 13. XGBoost ile Zaman Serisi Tahmini](13_xgboost_ile_zaman_serisi_tahmini.md) · [15. Derin Öğrenme ile Tahmin: LSTM, GRU ve 1D-CNN →](15_derin_ogrenme_ile_tahmin_lstm_gru_ve_1d_cnn.md)

## 14. Weka Zaman Serisi Tahmin Modülü (Forecast Sekmesi)

Bölüm 13.5'te Weka `Explorer` içindeki `Classify` sekmesini kullanarak işin mutfağını gördük: gecikme özelliklerini `TSLagMaker` filtresiyle elle ürettik ve bir regresyon algoritmasıyla test dönemini tahmin ettik. Bu yolun iki eksiği vardı: veri setinin bittiği tarihten sonrasını (ör. 1961 yılını) tahmin etmek zahmetliydi ve modelin "1 ay sonrası" ile "12 ay sonrası" için ne kadar başarılı olduğunu ayrı ayrı göremiyorduk.

Bu bölümde her iki sorunu da çözen **`Forecast`** sekmesini inceleyeceğiz. Bu sekme, Bölüm 13.5.1'de kurduğumuz `timeseriesForecasting` paketiyle birlikte `Explorer` penceresine eklenir. Forecast sekmesi gecikme ve takvim özelliklerini kendisi üretir, seçtiğiniz algoritmayla **özyinelemeli** çok adımlı tahmin yapar ve modelin başarısını her tahmin ufku için ayrı ayrı raporlar. Gecikmeleri arka planda kendisi ürettiği için 13.5.3'teki kopyalama, yeniden adlandırma ve sıralama adımlarına burada gerek yoktur.

### 14.1. Veri Hazırlığı: İki Farklı Yöntem

Veriyi hazırlamanın iki yolu vardır; ikisini de bilmenizde fayda var.

#### 14.1.1. Yöntem A: Dosya Yüklerken Ayarlama (Invoke Options)

Veriyi yükleme aşamasında Weka'ya "bu sütun tarihtir" diyebiliriz. Bu yol, sonradan dosya düzenlemekle uğraşmaktan daha temizdir.

1. `Explorer` penceresinin `Preprocess` sekmesinde **`Open file...`** düğmesine basın.
2. Dosya seçim penceresinde CSV dosyanızı seçin, **ancak hemen `Open` demeyin.**
3. Pencerenin altındaki **`Invoke options dialog`** kutucuğunu işaretleyin.
4. Şimdi `Open` deyin. Karşınıza CSV yükleyicisinin ayar penceresi gelecektir.
5. Bu pencerede şu iki satırı bulup değiştirin:
   - **`dateAttributes`**: Tarih sütununun sıra numarası (AirPassengers için **`1`**).
   - **`dateFormat`**: Dosyadaki tarih biçimi, harfiyen (AirPassengers için **`yyyy-MM`**; `yyyy` dört haneli yıl, büyük `MM` iki haneli ay demektir; küçük `mm` ise dakika anlamına gelir).
6. `OK` dediğinizde veri seti, tarih sütunu `Date` tipine dönüşmüş olarak açılır. Sol alttaki `Attributes` listesinde sütuna tıkladığınızda sağ panelde `Type: Date` yazdığını görmelisiniz.
7. **Çok önemli son adım:** Üstteki `Edit...` düğmesine basın. `Month` sütununun başlığına sağ tıklayıp `Rename attribute` seçeneğiyle adını **`Tarih`** olarak değiştirin ve `OK` ile kaydedin. Forecast sekmesi aylık veride kendisi de `Month` adında bir takvim sütunu ürettiği için bu değişikliği yapmazsak ad çakışması nedeniyle hata alırız.

> **Hemen deneyin 14.1.1 · Yükledikten sonra Preprocess panelini okuyun** (Weka Explorer, ~3 dk)
>
> Yukarıdaki yedi adımla `AirPassengers.csv` dosyasını `Invoke options dialog` üzerinden (`dateAttributes` = `1`, `dateFormat` = `yyyy-MM`) yükleyin ve tarih sütununun adını `Tarih` yapın. Sonra `Preprocess` sekmesinin sol altındaki `Attributes` listesinde önce `Tarih`'e, sonra `Passengers`'a tıklayıp sağ paneldeki bilgileri okuyun.
>
> **Ne göreceksiniz:** `Tarih` için `Type: Date`, `Missing: 0`, `Distinct: 144`, `Unique: 144 (100%)`: her ay bir kez geçer. `Passengers` için `Type: Numeric`, `Distinct: 118`, `Unique: 93 (65%)` ve istatistik tablosunda `Minimum` 104, `Maximum` 622, `Mean` 280.299, `StdDev` 119.966. Bu değerler Gretl'in `summary` çıktısıyla (Bölüm 11.3.1) aynıdır.
>
> **Şimdi siz:** (1) 144 aylık veride `Passengers` neden yalnızca 118 farklı (`Distinct`) değer alıyor; `Unique: 93` ne anlama geliyor? (2) Dosyayı bir kez de `Invoke options dialog` kutusunu işaretlemeden açın: `Month` sütununun `Type` satırında ne yazar?
>
> <details><summary>Cevap</summary>
>
> (1) Bazı yolcu sayıları birden fazla ayda tekrarlanır (ör. 1949'un Temmuz ve Ağustos aylarının ikisinde de 148); bu yüzden farklı değer sayısı 118'dir. `Unique: 93`, yalnızca bir kez geçen değerlerin sayısıdır. (2) `Type: Nominal` yazar ve `Distinct: 144` görünür: Weka her tarihi ayrı bir kategori (etiket) olarak okumuştur, sıralı bir zaman olarak değil. Forecast sekmesi bu sütunu zaman damgası olarak kullanamaz.
>
> </details>

#### 14.1.2. Yöntem B: ARFF Dosyasını Düzenleyerek Dönüştürme

Dosyayı doğrudan (seçenek penceresi olmadan) yüklediyseniz tarih sütunu nominal olarak okunur. Weka'nın standart kurulumunda nominal bir sütunu doğrudan tarihe çeviren hazır bir filtre bulunmaz; bunun yerine veriyi Weka'nın kendi dosya biçimi olan **ARFF**'e kaydedip sütun tanımını bir metin düzenleyicide değiştirebiliriz. ARFF dosyasının başında her sütun bir `@attribute` satırıyla tanımlanır; altında `@data` satırından sonra veriler gelir.

1. **ARFF olarak kaydetme:** `Preprocess` sekmesinde üstteki `Save...` düğmesine basın ve dosyayı örneğin `AirPassengers.arff` adıyla kaydedin.
2. **Tanımı değiştirme:** Dosyayı Not Defteri gibi bir metin düzenleyiciyle açın. `@attribute Month {1949-01,1949-02,...}` diye başlayan uzun satırı (nominal sütunun bütün değerlerini sayan satır) tümüyle silin ve yerine şu satırı yazın:

   ```text
   @attribute Tarih date yyyy-MM
   ```

   Bu tek satır hem sütunun adını `Tarih` yapar hem de tipini `yyyy-MM` biçimli tarih olarak tanımlar. `@data` altındaki `1949-01,112` gibi satırlara dokunmayın; bu değerler yeni tanımla tarih olarak okunur.
3. **Yeniden açma:** Dosyayı kaydedip Weka'da `Open file...` ile açın. `Tarih` sütununa tıkladığınızda `Type: Date` yazdığını görmelisiniz.

> **Hemen deneyin 14.1.2 · ARFF dosyasını sıfırdan elle yazın** (Not Defteri ve Weka Explorer, ~4 dk)
>
> Not Defteri'ne aşağıdaki satırları yazın ve **Farklı Kaydet** ile `satis.arff` adıyla kaydedin (Kayıt türü: **Tüm dosyalar**):
>
> ```text
> @relation satis
>
> @attribute Tarih date yyyy-MM
> @attribute Satis numeric
>
> @data
> 2024-01,120
> 2024-02,135
> 2024-03,128
> 2024-04,150
> 2024-05,162
> 2024-06,171
> ```
>
> Weka'da `Preprocess` → **`Open file...`** ile dosyayı açın (dosya türü olarak ARFF seçili olsun).
>
> **Ne göreceksiniz:** Üstte `Instances: 6`, `Attributes: 2` yazar. `Tarih`'e tıklayınca `Type: Date`, `Distinct: 6` görünür. Yani 14.1.2'deki tek satırlık tanım, kendi verinizde de tarih sütununu doğrudan `Date` tipinde açar.
>
> **Şimdi siz:** `@attribute Tarih date yyyy-MM` satırını `@attribute Tarih date yyyy-MM-dd` yapıp kaydedin ve dosyayı yeniden açın. `@data` satırlarına dokunmayın. Ne olacağını tahmin edin.
>
> <details><summary>Cevap</summary>
>
> Dosya açılmaz; Weka bir hata penceresi gösterir. Mesajda `unparseable date: 2024-01` ifadesi ve sorunun bulunduğu satır numarası (`line 7`, yani ilk veri satırı) yer alır. Tanımdaki biçim gün de beklerken veride gün yoktur. `dateFormat` (ya da ARFF'teki biçim) dosyadaki tarih yazımıyla harfiyen aynı olmalıdır.
>
> </details>

---

### 14.2. Forecast Sekmesi: Temel Ayarlar (Basic Configuration)

Verimiz hazırsa `Forecast` sekmesine geçelim. `Basic configuration` alt sekmesinde şu ayarları yapın:

1. **Target selection (tahmin hedefi):** Listeden `Passengers` sütununu işaretleyin.
2. **Time stamp (zaman damgası):** `Tarih` sütununu seçin.
3. **Periodicity (periyot):** **`Monthly`** seçin. Bunu seçtiğimizde Weka ay ve çeyrek özelliklerini (`Month`, `Quarter`) otomatik olarak ekler ve gecikme aralığını 1–12 ay olarak ayarlar.
4. **Number of time units to forecast (tahmin edilecek adım sayısı):** **`12`** yazın. Bu, verinin bittiği tarihten sonraki 12 ay, yani 1961 yılının tamamı için tahmin istediğimiz anlamına gelir. Aynı sayı, değerlendirmede hangi ufka kadar hata hesaplanacağını da belirler (bkz. 14.5).
5. **Perform evaluation:** Bu kutucuğun işaretli olduğundan emin olun; aksi hâlde yalnızca tahmin üretilir, başarı ölçülmez. Bu kutu işaretlendiğinde Weka, `Advanced configuration` altındaki `Evaluate on training` seçeneğini de kendiliğinden işaretler; bunu 14.3.5'te düzelteceğiz.

#### 14.2.1. Arka Planda Ne Olur? Özyinelemeli (Recursive) Tahmin

Forecast sekmesinin içindeki algoritma (ör. `LinearRegression` ya da `RandomForest`) aslında yalnızca **bir adım ilerisini** tahmin etmeyi bilir: son 12 ayın değerlerini ve takvim bilgisini alır, bir sonraki ayın değerini üretir. Peki 12 ay ilerisini nasıl tahmin eder?

**Açıklama:** Weka, Bölüm 13.5.6'da elle yapmanın ne kadar zahmetli olduğunu gördüğümüz işlemi otomatik yapar. Ocak 1961 tahmini üretildikten sonra bu tahmin, sanki gerçekleşmiş bir değermiş gibi girdi penceresine eklenir, penceredeki en eski değer dışarı atılır ve Şubat 1961 tahmin edilir. Aynı işlem istenen adım sayısına ulaşılana kadar tekrarlanır.

**Tanım (Özyinelemeli çok adımlı tahmin):** Bir adım ileri tahmin yapan model $f$, son gözlem zamanı $T$ ve kullanılan gecikme sayısı $p$ olmak üzere, $h$ adım ilerideki tahmin şöyle üretilir:

$$\hat{y}_{T+h} = f(\tilde{y}_{T+h-1}, \tilde{y}_{T+h-2}, \dots, \tilde{y}_{T+h-p})$$

Burada girdideki her değer, gözlenmişse gerçek değerin kendisi, henüz gözlenmemişse modelin önceki adımda ürettiği tahmindir: $s \le T$ için $\tilde{y}_ s = y_s$, $s > T$ için $\tilde{y}_ s = \hat{y}_ s$.

> **Simge notu:** $`T`$: son gözlemin zamanı (AirPassengers'ta Aralık 1960) · $`h`$: kaç adım ileriye tahmin yapıldığı · $`p`$: modelin geriye baktığı gecikme sayısı (burada 12) · $`\hat{y}_{T+h}`$ *(y şapka, T artı h)*: T anından h adım ilerisi için üretilen tahmin · $`\tilde{y}_s`$ *(y tilda, s)*: s anı için girdide kullanılan değer (gerçek ya da tahmin); dalgalı çizgi (tilda) "bu değer gerçek de olabilir, tahmin de" demektir · $`s`$: girdideki herhangi bir ay · $`\dots`$ *(üç nokta)*: aradaki terimler · $`\le`$ *(küçük eşittir)*

Formülü $p = 12$ ile somutlaştıralım:

- $h = 1$ (Ocak 1961): Girdiler Ocak–Aralık 1960'tır; 12 değerin **12'si de gerçektir**.
- $h = 2$ (Şubat 1961): Girdiler Şubat–Aralık 1960 (11 gerçek değer) ile Ocak 1961 tahminidir (1 tahmin).
- $h = 12$ (Aralık 1961): Girdiler Aralık 1960 (1 gerçek değer) ile Ocak–Kasım 1961 tahminleridir (11 tahmin).

![Özyinelemeli çok adımlı tahmin](../images/ch14_ozyinelemeli_tahmin.svg)

*Şekil 14.1 — Özyinelemeli tahmin. Her adımda model yalnızca bir adım ilerisini tahmin eder (turuncu); bu tahmin bir sonraki adımın girdi penceresine eklenir ve en eski değer pencereden çıkar. Üçüncü adımda girdilerin üçte ikisi artık modelin kendi tahminleridir.*

**Yorum:** Bu yöntemin doğal bir sonucu **hata birikimidir**. 1 adım ileri tahminde bütün girdiler gerçek değerdir. 12 adım ileri tahminde ise yukarıdaki örnekte görüldüğü gibi 12 girdinin 11'i modelin kendi (hatalı olabilecek) tahminleridir; ilk adımlarda yapılan küçük bir hata sonraki adımlara taşınır ve büyüyebilir. Bu yüzden uzun ufuklu tahminlerin hatasını ayrıca incelemek gerekir (14.5).

**Not —** Ağaç tabanlı bir temel öğrenici (ör. `RandomForest`) seçerseniz, Bölüm 13.3'teki ekstrapolasyon sınırlılığı burada da geçerlidir: özyinelemeli tahmin trendi eğitimde görülen düzeyin üzerine taşıyamaz. Bunu 14.3.1'deki alıştırmada ve 14.4'teki çıktıda sayılarla göreceksiniz. Forecast sekmesinin trend ayarlaması (`Lag creation` sekmesindeki zaman indeksi özellikleri: zamanın kendisi, karesi, küpü ve gecikmelerle çarpımı) bu sorunu doğrusal öğrenicilerde (ör. `LinearRegression`, doğrusal çekirdekli `SMOreg`) kısmen çözer; ağaçlarda ise çözmez.

> **Hemen deneyin 14.2.1 · Ufku 12'den 24'e çıkarın: 1961 tahminleri değişir mi?** (Weka Explorer, ~4 dk)
>
> `Forecast` sekmesinde 14.2'deki temel ayarları yapın (`Passengers`, `Tarih`, `Monthly`, `12`, `Perform evaluation`), `Evaluation` sekmesinde `Evaluate on training` işaretini kaldırıp `Evaluate on held out training` = `0.1` seçin, `Base learner` varsayılan `LinearRegression` kalsın. **`Start`** deyin ve `=== Future predictions from end of test data ===` bölümünde `1961-01*` ile `1961-07*` değerlerini not edin. Sonra `Number of time units to forecast` kutusunu **`24`** yapıp yeniden `Start` deyin.
>
> **Ne göreceksiniz:** İlk çalıştırmada `1961-01*` = 412.4956 ve `1961-07*` = 679.8498'dir. 24 adımlık çalıştırmada liste `1962-12*`'ye kadar uzar, ama 1961'in 12 değeri **birebir aynı** kalır (yine 412.4956 ve 679.8498). Özyinelemeli tahminde Ocak 1961, Şubat 1961, … sırayla üretilir; 1962 tahminleri 1961 tahminlerinin üzerine eklenir, öncekileri değiştirmez.
>
> **Şimdi siz:** `1962-07*` ve `1962-12*` değerlerini okuyun. Temmuz ile Aralık arasındaki farkı 1960 (gerçek: 622 ve 432), 1961 ve 1962 için hesaplayın. Bu fark nasıl değişiyor ve bu makul mü?
>
> <details><summary>Cevap</summary>
>
> `1962-07*` = 791.3482, `1962-12*` = 314.6953. Temmuz–Aralık farkı 1960'ta 622 − 432 = 190, 1961 tahmininde 679.8 − 386.5 ≈ 293, 1962 tahmininde 791.3 − 314.7 ≈ 477'dir. Mevsimsel salınım her yıl gerçekçi olmayan biçimde büyür; Aralık 1962 tahmini Aralık 1960'ın gerçek değerinin bile çok altındadır. Bu, 14.2.1'de anlatılan hata birikiminin somut bir örneğidir: ufuk uzadıkça model giderek kendi tahminleri üzerine tahmin yapar. Ayrıca değerlendirme tablosu artık 24 sütunludur; 15. adımdan sonraki sütunlarda `N` = 0 ve hata `N/A` yazar (nedeni 14.6'da).
>
> </details>

---

### 14.3. Gelişmiş Ayarlar (Advanced Configuration): Sekme Sekme İnceleme

Şimdi `Advanced configuration` alt sekmesine geçin. Burada altı ayrı sekme göreceksiniz: `Base learner`, `Lag creation`, `Periodic attributes`, `Overlay data`, `Evaluation` ve `Output`. Aşağıdaki ayarları sırasıyla yapın.

#### 14.3.1. Base Learner (Temel Öğrenici)

Tahmin algoritmasının seçildiği yerdir. Varsayılan `LinearRegression` basit kalabilir. `Choose` düğmesiyle **`functions → SMOreg`** veya **`trees → RandomForest`** seçebilirsiniz. Gradient boosting denemek isterseniz **`meta → AdditiveRegression`** (Bölüm 13.5.4) da seçilebilir. Seçtiğiniz algoritmanın ayarlarını, algoritma adının yazılı olduğu kutuya tıklayarak değiştirebilirsiniz.

> **Hemen deneyin 14.3.1 · Üç temel öğreniciyi yarıştırın** (Weka Explorer, ~6 dk)
>
> 14.2.1'deki ayarlarla (ufuk `12`, held out `0.1`) çalışın. `Base learner` sekmesinde sırasıyla `LinearRegression` (varsayılan), **`functions → SMOreg`** ve **`trees → RandomForest`** seçin; algoritma ayarlarına dokunmadan her birinde `Start` deyin. Her çalıştırmada üç sayıyı aşağıdaki tabloya yazın: `=== Evaluation on test data ===` tablosunda `Mean absolute error` satırının `1-step-ahead` ve `12-steps-ahead` sütunları ile `1961-07*` tahmini.
>
> | Öğrenici | MAE (1 adım) | MAE (12 adım) | Temmuz 1961 tahmini |
> | --- | --- | --- | --- |
> | LinearRegression | 19.7823 | 32.3801 | 679.8498 |
> | SMOreg | ? | ? | ? |
> | RandomForest | ? | ? | ? |
>
> **Ne göreceksiniz:** `LinearRegression` satırı yukarıdaki gibidir. Diğer iki satırı siz dolduracaksınız.
>
> **Şimdi siz:** Tabloyu doldurun. 1 adım ileride hangisi en iyi? Hangisinin 1961 tahmini, 1960 Temmuz'unun gerçek değeri olan 622'nin altında kalıyor ve neden?
>
> <details><summary>Cevap</summary>
>
> | Öğrenici | MAE (1 adım) | MAE (12 adım) | Temmuz 1961 tahmini |
> | --- | --- | --- | --- |
> | LinearRegression | 19.7823 | 32.3801 | 679.8498 |
> | SMOreg | 15.4676 | 7.855 | 659.9748 |
> | RandomForest | 31.8779 | 17.4125 | 514.9335 |
>
> 1 adım ileride en iyi `SMOreg`, en kötü `RandomForest`'tır. `RandomForest`'ın Temmuz 1961 tahmini (514.9) 622'nin çok altında kalır: ağaçlar eğitimde gördükleri düzeyin üstüne çıkamaz (14.2.1'deki not, Bölüm 13.3). 12 adım sütunundaki küçük değerlere (7.855, 17.4125) güvenmeyin: bu sütun yalnızca 3 tahmine dayanır (`N` = 3, bkz. 14.6).
>
> </details>

#### 14.3.2. Lag Creation (Gecikme Oluşturma)

Modelin geçmişe ne kadar bakacağını belirleyen ayardır. Ana ekranda `Periodicity: Monthly` seçtiğimiz için Weka gecikmeleri zaten 1'den 12'ye kadar ayarlamıştır. Bu değerleri değiştirmek ya da açıkça görmek isterseniz:

- **Use custom lag lengths:** İşaretleyin.
- **Minimum lag:** `1` olarak bırakın.
- **Maximum lag:** **`12`** yapın. Mevsimselliği yakalamak için modelin bir yıl geriye bakması gerekir; 12. gecikme "geçen yılın aynı ayı" bilgisini taşır.

Aynı sekmedeki `Include powers of time` (zamanın karesi ve küpü) ve `Included products of time and lagged variables` (zaman × gecikme çarpımları) seçenekleri varsayılan olarak açıktır; bunlar 13.5.3'te gördüğümüz trend sütunlarını üretir.

> **Hemen deneyin 14.3.2 · Gecikme uzunluğu: 6, 12 ya da 24 ay?** (Weka Explorer, ~5 dk)
>
> 14.2.1'deki ayarlarla (`LinearRegression`, ufuk `12`, held out `0.1`) çalışın. `Lag creation` sekmesinde **`Use custom lag lengths`** kutusunu işaretleyin, `Minimum lag` = `1`, **`Maximum lag` = `6`** yapıp `Start` deyin. Test tablosundaki `Mean absolute error` satırının `1-step-ahead` ve `8-steps-ahead` sütunlarını ve `1961-07*` tahminini not edin.
>
> **Ne göreceksiniz:** `Maximum lag` = 6 ile 1 adım MAE 18.9307, 8 adım MAE 36.5292 ve `1961-07*` = 586.248'dir. Varsayılan 12 gecikmede bu değerler 19.7823, 24.5711 ve 679.8498'di. Kısa vadede fark yoktur, ama 6 gecikmeli model "geçen yılın aynı ayı" bilgisini (12. gecikme) göremediği için orta vadede hatası büyür ve Temmuz 1961 tahmini, Temmuz 1960'ın gerçek değeri 622'nin bile altında kalır.
>
> **Şimdi siz:** `Maximum lag` değerini **`24`** yapın. 2 yıl geriye bakmak sonuçları belirgin biçimde iyileştirir mi? Önce tahmin edin.
>
> <details><summary>Cevap</summary>
>
> `Maximum lag` = 24 ile 1 adım MAE 19.7544, 8 adım MAE 22.4536, `1961-07*` = 694.8721 çıkar. 12 gecikmeye göre değişim küçüktür (19.78 → 19.75 ve 24.57 → 22.45), buna karşılık model iki kat fazla gecikme sütunu (ve zaman × gecikme çarpımı) kullanır. Mevsim periyodu 12 olan aylık veride `Maximum lag` = 12 hem yeterli hem sade bir seçimdir.
>
> </details>

#### 14.3.3. Periodic Attributes (Periyodik Özellikler)

Ana ekranda `Periodicity: Monthly` seçtiğimiz için Weka ay ve çeyrek özelliklerini zaten otomatik ekler. Bu sekmedeki `Customize` kutusu işaretlenerek özel tatil günleri gibi ek takvim özellikleri tanımlanabilir; bizim örneğimizde müdahale etmenize gerek yoktur.

> **Hemen deneyin 14.3.3 · Takvim özelliklerini kapatın** (Weka Explorer, ~5 dk)
>
> Etkiyi açıkça görmek için gecikmeleri kısaltalım: 14.3.2'deki gibi `Use custom lag lengths` işaretli, **`Maximum lag` = `3`** olsun (`LinearRegression`, ufuk `12`, held out `0.1`). Önce takvim özellikleri açıkken (varsayılan) `Start` deyin. Sonra `Periodic attributes` sekmesinde **`Customize`** kutusunu işaretleyin ve açılan listede **hiçbir kutunun** (`Month`, `Quarter` dahil) işaretli olmadığından emin olun; yeniden `Start` deyin. İki çalıştırmada 1 adım MAE'yi ve 1961'in Ocak ve Temmuz tahminlerini karşılaştırın.
>
> **Ne göreceksiniz:** Takvim özellikleri açıkken 1 adım MAE 20.39, `1961-01*` = 454.803, `1961-07*` = 574.4533'tür: tahminlerde yaz zirvesi vardır. Kapatınca 1 adım MAE **32.0144**'e çıkar ve 1961 tahminleri neredeyse düz bir çizgiye döner (`1961-01*` = 473.5462, `1961-07*` = 506.2319; bütün yıl 473 ile 534 arasında). Son 3 ayı gören bir model, `Month` sütunu olmadan "Temmuz'da yolcu artar" bilgisine ulaşamaz.
>
> **Şimdi siz:** `Maximum lag` değerini yeniden `12` yapın, takvim özellikleri yine kapalı kalsın. Temmuz 1961 tahmini ve 1 adım MAE ne olur? Fark neden bu kez daha küçüktür?
>
> <details><summary>Cevap</summary>
>
> `1961-07*` = 604.9814 ve 1 adım MAE 21.9257 çıkar (takvim özellikleri açıkken 679.8498 ve 19.7823'tü). Kayıp küçüktür, çünkü 12. gecikme "geçen yılın aynı ayı" bilgisini zaten taşır; mevsimsellik gecikmeler üzerinden kısmen yakalanır. Kısa gecikmeli modellerde ise mevsimselliği taşıyan tek şey takvim özellikleridir.
>
> </details>

#### 14.3.4. Overlay Data (Dış Değişkenler)

Tahmini etkileyebilecek dış değişkenlerin (döviz kuru, akaryakıt fiyatı vb.) tanımlandığı yerdir (`Use overlay data` kutusu). Model bu değişkenlerin **aynı aydaki** değerini girdi olarak kullanır. Dolayısıyla geleceği tahmin ederken bu değişkenlerin gelecekteki değerlerinin de bilinmesi gerekir: Örneğin 1961 için tahmin istiyorsanız veri dosyasına 1961'in 12 ayını eklemeli, akaryakıt fiyatı sütununu (plan ya da beklenti değerleriyle) doldurmalı, yolcu sayısını ise boş (`?`) bırakmalısınız. Dış veri kullanmadığımız için burayı boş geçiyoruz.

> **Hemen deneyin 14.3.4 · Dış değişken eklemek her zaman işe yarar mı?** (Weka Explorer, ~6 dk)
>
> Bölüm 10'daki `data/macro.csv` dosyasını 14.1.1'deki gibi `Invoke options dialog` ile açın: `dateAttributes` = `1`, `dateFormat` = **`yyyy-MM-dd`** (bu dosyada tarih sütununun adı `date` olduğu için yeniden adlandırmaya gerek yoktur). `Forecast` sekmesinde:
>
> 1.  `Target selection` listesinde **yalnızca `inflation`**'ı işaretleyin; `Time stamp` = `date`, `Periodicity` = `Monthly`, `Number of time units to forecast` = **`3`**.
> 2.  `Evaluation` sekmesinde `Evaluate on training` işaretini kaldırın, `Evaluate on held out training` = **`12`** yapın; `Base learner` `LinearRegression` kalsın. `Start` deyin.
> 3.  Sonra `Overlay data` sekmesinde **`Use overlay data`** kutusunu işaretleyip listeden yalnızca **`exchange`**'i seçin ve yeniden `Start` deyin.
>
> **Ne göreceksiniz:** Dış değişken olmadan test tablosunda MAE 1, 2 ve 3 adım için 6.7529, 15.558 ve 28.2149'dur ve 2025'in ilk üç ayı için tahmin listelenir. `exchange` eklenince MAE **16.2861, 31.857 ve 47.0238**'e çıkar ve gelecek tahmini bölümünde tahmin yerine şuna benzer bir uyarı görünür: `Unable to generate future forecast because there is no future overlay data available.`
>
> **Şimdi siz:** Uyarının nedenini kendi cümlenizle açıklayın. 2025 tahminini alabilmek için veri dosyasına ne eklemeniz gerekirdi?
>
> <details><summary>Cevap</summary>
>
> Model enflasyonu aynı aydaki döviz kuruyla açıklar; 2025 Ocak enflasyonunu tahmin etmek için 2025 Ocak kurunun bilinmesi gerekir ve dosyada bu değer yoktur. Tahmin alabilmek için dosyaya 2025-01-01, 2025-02-01 ve 2025-03-01 satırlarını eklemeli, `exchange` sütununa varsayımsal kur değerlerini yazmalı, `inflation` (ve kullanmıyorsanız `interest`) için `?` bırakmalısınız. Bu veride ve bu ayarlarla dış değişken test hatasını azaltmamış, aksine artırmıştır: bir değişkenin modele eklenmesi, tahmini kendiliğinden iyileştirmez; etkisi her zaman held out test hatasıyla kontrol edilmelidir.
>
> </details>

#### 14.3.5. Evaluation (Değerlendirme)

Modelin başarısının nerede ve nasıl ölçüleceği burada ayarlanır. Bu sekme, sonuçların güvenilirliğini doğrudan belirlediği için en dikkatli ayarlanması gereken sekmedir.

**Sağ taraftaki test seçenekleri:**

- **Evaluate on training (eğitim verisiyle test et):** `Perform evaluation` işaretlendiğinde bu kutu kendiliğinden işaretlenir; **işaretini kaldırın.** Bu, soruları önceden gören bir öğrencinin sınava girmesi gibidir. Model eğitim verisini ezberlemiş olabilir (aşırı öğrenme); hata olduğundan çok düşük görünür, ama gerçek gelecekte model başarısız olabilir.
- **Evaluate on held out training (ayrılmış veriyle test et):** **İşaretleyin.** Yanındaki kutuya ya bir gözlem sayısı (ör. **`12`**) ya da 1'den küçük bir oran (ör. `0.1`, verinin %10'u) yazılır; kutudaki varsayılan değer `0.3`'tür. `0.1` yazarsanız Weka eğitim kısmını $144 \cdot 0.9 = 129.6 \approx 130$ ay olarak yuvarlar ve son **14 ayı** (Kasım 1959 – Aralık 1960) saklar.
  - **Mantığı:** Weka serinin son kısmını (ör. son 12 ayı) eğitimden çıkarıp saklar, modeli geri kalan veriyle eğitir, sonra saklanan dönemi tahmin ederek gerçek değerlerle karşılaştırır. Gerçekçi başarı testi budur (eğitim/test ayrımının mantığı için bkz. Bölüm 8).

**Sol taraftaki metrik listesi (`Metrics`):** Başarının hangi ölçütlerle raporlanacağını buradan seçersiniz. Varsayılan olarak işaretli olan şu ikisinin seçili kaldığından emin olun (tanımları için bkz. Bölüm 8):

- **Mean absolute error (MAE)**
- **Root mean squared error (RMSE)**

İsterseniz ölçekten bağımsız karşılaştırma için **Mean absolute percentage error (MAPE)** da işaretleyebilirsiniz.

> **Hemen deneyin 14.3.5 · Eğitim hatası mı, test hatası mı?** (Weka Explorer, ~4 dk)
>
> AirPassengers ve 14.2.1'deki ayarlarla (`LinearRegression`, ufuk `12`) çalışın, ama bu kez `Evaluation` sekmesinde **hem `Evaluate on training` hem `Evaluate on held out training` (`0.1`)** işaretli olsun. `Start` deyin. Çıktıda iki tablo görünür: `=== Evaluation on training data ===` ve `=== Evaluation on test data ===`. İkisinde de `1-step-ahead` sütunundaki `N` ve `Mean absolute error` değerlerini okuyun.
>
> **Ne göreceksiniz:** Eğitim tablosunda `N` = 118 ve MAE **7.8316**; test tablosunda `N` = 14 ve MAE **19.7823**'tür. Model, eğitimde gördüğü ayları 2.5 kattan daha isabetli "tahmin eder". Eğitim tablosunda `N`'nin 130 değil 118 olması, ilk 12 ayın gecikme değerleri (önceki 12 ay) bulunmadığı için hesaba katılamamasındandır.
>
> **Şimdi siz:** `Base learner`'ı `RandomForest` yapıp aynı karşılaştırmayı yapın. Eğitim ve test MAE'leri arasındaki fark büyür mü, küçülür mü?
>
> <details><summary>Cevap</summary>
>
> `RandomForest`'ta eğitim MAE'si **5.6887** (üç öğrenici arasında en düşük), test MAE'si ise **31.8779**'dur (en yüksek); test hatası eğitim hatasının yaklaşık 5.6 katıdır. Eğitim verisine en iyi uyan model, yeni verideki en kötü model çıkmıştır: `Evaluate on training` sonucuna bakarak model seçmek bu yüzden yanıltıcıdır.
>
> </details>

#### 14.3.6. Output (Çıktı Ayarları)

`Start` düğmesine bastıktan sonra karşımıza ne çıkacağı burada belirlenir.

**Sol panel (çıktı seçenekleri):**

- **Output predictions at step:** İşaretleyin ve yanındaki adım değerini (`Step to output`) `1` bırakın. Böylece test için ayırdığımız dönemin 1 adım ileri tahminlerini gerçek değerlerle birlikte sayısal döküm olarak görebiliriz.
- **Output future predictions beyond end of series:** **En önemli ayar budur;** varsayılan olarak işaretlidir, işaretli kaldığından emin olun. İşaret kaldırılırsa veri setinin bittiği tarihten sonraki (1961 yılı) tahminleri göremezsiniz.

**Sağ panel (grafik seçenekleri):**

- **Graph predictions at step:** İşaretleyin. Seçtiğiniz adımdaki (ör. 1 adım ileri) tahminleri, gerçek değerlerle birlikte aynı grafikte çizer; tahmin ile gerçek çizgilerin ne kadar üst üste bindiğini gözle görmenizi sağlar.
- **Graph target at steps:** İşaretlerseniz tek bir hedefin birden fazla ufuktaki tahminlerini (ör. `Steps to graph` kutusuna `1,3,12` yazarak 1, 3 ve 12 adım ileri tahminleri) gerçek değerlerle birlikte aynı grafikte görürsünüz. Ufuk uzadıkça tahmin çizgisinin gerçekten nasıl uzaklaştığını görmek için kullanışlıdır (14.5).
- **Graph future predictions beyond end of series:** Varsayılan olarak işaretlidir; 1961 tahminlerinin grafikte görünmesi için işaretli kalmalıdır.

> **Hemen deneyin 14.3.6 · Aynı ayı 1 ve 12 adım önceden tahmin etmek** (Weka Explorer, ~5 dk)
>
> 14.2.1'deki ayarlarla (`LinearRegression`, ufuk `12`, held out `0.1`, `Evaluate on training` kapalı) çalışın. `Output` sekmesinde `Output predictions at step` kutusunu işaretleyip `Step to output` değerini **`12`** yapın; sağ panelde **`Graph target at steps`** kutusunu işaretleyip `Steps to graph` kutusuna **`1,12`** yazın. `Start` deyin; metin çıktısında `=== Predictions for test data: Passengers (12-steps ahead) ===` başlığını bulun ve grafik panelindeki test grafiğine bakın.
>
> **Ne göreceksiniz:** Tabloda gerçek değeri bilinen yalnızca üç satır vardır: gerçek değerler 461, 390, 432 (Ekim–Aralık 1960), tahminler 430.4589, 371.2436, 384.1571, hatalar −30.5411, −18.7564, −47.8429. (Altında gerçek değeri `?` olan, test döneminin dışına taşan satırlar da görünebilir.) Bu üç hatanın mutlak değerlerinin ortalaması 32.38'dir: test tablosundaki `12-steps-ahead` MAE (32.3801) tam olarak budur. Grafikte 1 adım ileri tahmin çizgisi Kasım 1959'dan Aralık 1960'a kadar 14 noktadan oluşurken, 12 adım ileri tahmin çizgisi yalnızca Ekim–Aralık 1960'taki 3 noktadan oluşur.
>
> **Şimdi siz:** `Step to output` değerini `1` yapıp yeniden çalıştırın. Aynı üç ayın (gerçek 461, 390, 432) 1 adım ileri tahminleri ve hataları nedir? Bu üç ay için ortalama mutlak hatayı 12 adımlık 32.38 ile karşılaştırın.
>
> <details><summary>Cevap</summary>
>
> 1 adım ileri tahminler 424.9348, 389.0129, 382.0433; hatalar −36.0652, −0.9871, −49.9567. Ortalama mutlak hata (36.07 + 0.99 + 49.96) / 3 ≈ **29.00** olur; aynı aylar için 12 adım ileri tahminin 32.38'inden biraz küçüktür. Tek tek aylara bakınca 12 adım ileri tahmin bazen daha isabetli olabilir (Ekim ve Aralık'ta), ama ortalamada daha uzak ufuk daha büyük hata verir. Üç tahminlik bir ortalama da şansa çok açıktır (14.6).
>
> </details>

---

### 14.4. Sonuçların Okunması

Ayarları yaptıktan sonra `Start` düğmesine basın. Sonuçlar sağ taraftaki `Output` (metin) ve grafik panellerinde görünür.

**1. Grafik yorumu:** Grafiğin sağ tarafına odaklanın.

- **Test bölgesi:** Saklanan dönemde (held out = 12 ise 1960 yılı, 0.1 ise Kasım 1959 – Aralık 1960) iki çizgi görürsünüz: gerçek değerler ve tahminler. Birbirlerine yakınlıkları modelin başarısını gösterir. Özellikle yaz zirvelerinde tahminin gerçeğin altında kalıp kalmadığına bakın.
- **Gelecek bölgesi (1961):** Verinin bittiği noktadan sağa, boşluğa doğru uzanan tek çizgi, geleceğe dair özyinelemeli tahminimizdir. Bu bölgede karşılaştırılacak gerçek değer yoktur.

**2. Metin paneli yorumu:** Metin panelini kaydırarak şu başlıkları bulun (başlıkların tam yazımı Weka sürümüne göre küçük farklılıklar gösterebilir):

- **`=== Evaluation on test data ===`:** Ayrılmış (held out) veri üzerindeki değerlendirme sonuçlarıdır. 14.3.5'te seçtiğimiz **MAE** ve **RMSE** değerleri burada, her tahmin ufku için ayrı sütunlarda yer alır (ayrıntısı 14.5'te). Bu değerler ne kadar düşükse model o kadar başarılıdır.
- **`=== Future predictions from end of test data ===`:** 14.3.6'daki ayar sayesinde burada önce serinin son ayları, ardından **1961 yılının aylık yolcu tahminleri** listelenir. Tahmin edilen değerlerin yanında `*` işareti bulunur; bu işaret o satırın gerçek veri değil tahmin olduğunu gösterir. RandomForest ile (held out 0.1) aldığımız çıktıdan bir kesit:

```text
1960-11         390
1960-12         432
1961-01*    390.0954
1961-02*    404.0546
1961-03*     421.685
...
1961-07*    514.9335
1961-08*    517.6031
...
```

Sizin değerleriniz seçtiğiniz algoritmaya ve ayarlara göre farklı olacaktır; örneğin `LinearRegression` aynı ayarlarla Temmuz 1961 için yaklaşık 680 tahmin eder.

> **Hemen deneyin 14.4 · 1961 tahminlerini eleştirel gözle okuyun** (Weka Explorer, ~3 dk)
>
> 14.2.1'deki ayarlarla (`LinearRegression`, ufuk `12`, held out `0.1`) `Start` deyin. `=== Future predictions from end of test data ===` bölümünde 1961 yılının en büyük ve en küçük tahminini bulun ve 1960'ın gerçek değerleriyle (listede `*` işaretsiz satırlar) karşılaştırın.
>
> **Ne göreceksiniz:** 1961'in en büyük tahmini `1961-07*` = 679.8498 (1960 Temmuz'u 622), en küçüğü `1961-12*` = 386.4787'dir (`1961-11*` = 389.8774). Oysa 1960 Aralık'ının gerçek değeri 432'dir: büyüyen bir seride model, Aralık 1961 için bir önceki yıldan yaklaşık 46 bin daha az yolcu öngörüyor.
>
> **Şimdi siz:** `Base learner`'ı `SMOreg` yapın. Aralık 1961 tahmini kaç çıkar ve 432'nin üstünde midir? Bölüm 11'deki airline ARIMA modeli Aralık 1961 için yaklaşık 477 tahmin eder; hangi model ona daha yakındır?
>
> <details><summary>Cevap</summary>
>
> `SMOreg` ile `1961-12*` = 456.2058'dir; 1960 Aralık'ının (432) üstündedir ve ARIMA'nın tahminine (477.2) `LinearRegression`'dan (386.5) çok daha yakındır. Bir tahmin tablosunu yalnızca hata metrikleriyle değil, "bu değerler serinin bilinen davranışıyla (trend + mevsimsellik) uyumlu mu?" sorusuyla da okumak gerekir.
>
> </details>

---

### 14.5. Adım Adım Hata Analizi (Ufuk Testi)

`=== Evaluation on test data ===` başlığının altındaki tablo, yan yana uzanan geniş bir tablodur ve genellikle gözden kaçar. Oysa modelin güvenilirliğini, yani **kararlılığını** ölçen asıl yer burasıdır: tablo, modelin performansını tahmin ufkuna göre ayrı ayrı raporlar.

**Açıklama:** Bir modelin "gelecek ayı" tahmin etmesiyle "bir yıl sonrasını" tahmin etmesi aynı zorlukta değildir. 14.2.1'de gördüğümüz gibi, 1 adım ileri tahminde bütün girdiler gerçek değerlerdir; 12 adım ileri tahminde ise girdiler büyük ölçüde modelin kendi tahminlerinden oluşur ve hatalar birikir. Bu nedenle tahmin ufku uzadıkça hatanın artmasını bekleriz.

Tabloyu şöyle okumalısınız:

- **Sütunlar (`1-step-ahead` … `12-steps-ahead`):** Her sütun bir tahmin ufkunu ( $h$ ) gösterir.
  - **`1-step-ahead`:** Modelin 1 ay sonrasını tahmin ederken yaptığı hata.
  - **`12-steps-ahead`:** Modelin 12 ay (1 yıl) sonrasını tahmin ederken yaptığı hata.
- **Satırlar:** İlk satır `N` (14.6), sonraki satırlar 14.3.5'te seçtiğiniz metriklerdir.

Aşağıdaki değerler `RandomForest`, `Monthly` periyot, gecikme 1–12 ve held out = `0.1` (14 ay) ayarlarıyla elde edilmiştir. Weka bu tabloyu yatay basar; okumayı kolaylaştırmak için burada satır ve sütunları yer değiştirerek veriyoruz:

| Ufuk $`h`$ | N | MAE | RMSE |
| --- | --- | --- | --- |
| 1 | 14 | 31.9 | 44.3 |
| 2 | 13 | 35.0 | 46.6 |
| 3 | 12 | 36.4 | 48.2 |
| 4 | 11 | 35.9 | 48.7 |
| 5 | 10 | 38.9 | 51.1 |
| 6 | 9 | 43.2 | 54.1 |
| 7 | 8 | 45.3 | 57.0 |
| 8 | 7 | 48.5 | 60.3 |
| 9 | 6 | 46.7 | 60.4 |
| 10 | 5 | 34.5 | 45.2 |
| 11 | 4 | 20.5 | 23.0 |
| 12 | 3 | 17.4 | 19.3 |

- `1-step-ahead` sütununda MAE **31.9**'dur: Model bir sonraki ayı tahmin ederken ortalama yaklaşık 32 bin yolcu yanılıyor demektir (AirPassengers değerleri bin yolcu cinsindendir).
- `5-steps-ahead` sütununda MAE **38.9**'dur: 5 ay sonrasını tahmin ederken hata payı artmıştır; 8 adımda 48.5'e kadar çıkar.
- 10, 11 ve 12 adımda hata yeniden düşer (12 adımda 17.4). Bu, modelin uzak geleceği daha iyi bildiği anlamına **gelmez**: Bu sütunlardaki ortalamalar yalnızca 5, 4 ve 3 tahmine dayanır (N değeri, 14.6).

**Yorumlama mantığı:** Tabloyu şu üç duruma göre okuyun:

- Hata değerleri 1. aydan 12. aya doğru **çok hızlı artıyorsa**, model kısa vade için güvenilirdir ama uzun vadeli planlama (ör. gelecek yılın yatırım kararları) için risklidir.
- Hata değerleri **sabit kalıyor veya az artıyorsa**, model kararlı (stabil) ve güvenilir bir yapıdadır.
- Hata uzun ufuklarda **beklenmedik biçimde düşüyorsa**, önce `N` satırına bakın; az sayıda tahmine dayanan ortalamalar şansa çok açıktır.

**Özetle:** Raporlarınızda yalnızca tek bir genel hata değeri vermek yerine bu tabloya dayanarak *"Modelimiz ilk 3 ay için isabetli tahminler yapıyor, ancak 6. aydan sonra hata payı belirgin biçimde artıyor"* şeklinde ufka bağlı, ayrıntılı bir yorum yapabilirsiniz.

> **Hemen deneyin 14.5 · LinearRegression'ın ufuk tablosunu RandomForest'ınkiyle karşılaştırın** (Weka Explorer, ~4 dk)
>
> 14.2.1'deki ayarlarla (`LinearRegression`, ufuk `12`, held out `0.1`) `Start` deyin ve `=== Evaluation on test data ===` tablosundaki `Mean absolute error` satırını baştan sona okuyun. Yukarıdaki `RandomForest` tablosuyla karşılaştırın.
>
> **Ne göreceksiniz:** MAE değerleri 1'den 12'ye kadar şöyledir: 19.7823, 25.4258, 25.4049, 24.0274, 24.4563, 26.4726, 24.3351, 24.5711, 21.6989, 22.9647, 24.2034, 32.3801. 1. adımdan sonra hata 11. adıma kadar yaklaşık 22–26 aralığında kalır; `RandomForest`'ta ise 31.9'dan 8. adımda 48.5'e çıkıyordu. Yukarıdaki ölçütlere göre `LinearRegression` daha kararlıdır.
>
> **Şimdi siz:** 12. adımdaki ani sıçrama (24.2 → 32.4) "model 12. ayda çöküyor" diye yorumlanabilir mi? Tablodaki hangi satıra bakmalısınız?
>
> <details><summary>Cevap</summary>
>
> Hayır. `N` satırına bakın: 12 adım sütunu yalnızca 3 tahmine dayanır (Ekim–Aralık 1960; 14.3.6'daki üç satır). Aralık 1960'taki tek büyük hata (−47.8) bu ortalamayı tek başına yukarı çeker. Güvenilir bir yorum için test dönemini uzatmak gerekir (14.6'daki alıştırma).
>
> </details>

---

### 14.6. Tablodaki "N" Değeri ve Veri Sınırı

Tablonun en üstündeki **`N`** satırı, o ufuk için **kaç tahminin gerçek değerle karşılaştırılabildiğini**, yani hatanın kaç gözlem üzerinden hesaplandığını gösterir.

14.5'teki çıktıda `1-step-ahead` için **N = 14** iken `12-steps-ahead` için bu sayı **N = 3**'e düşer (held out oranı `0.1` seçildiği için 144 aylık verinin son 14 ayı saklanmıştır, bkz. 14.3.5). Bu düşüş bir hata değil, test verisinin sonlu olmasının doğal sonucudur.

**Açıklama:** Test için ayrılmış 14 aylık gerçek veri olduğunu düşünün. Weka, tahmine eğitim verisinin sonundan başlar ve test verisi boyunca birer ay ilerleyerek her noktadan yeniden tahmin yapar.

- **Kısa vade (1 ay sonrası):** Hangi noktadan başlarsanız başlayın, bir sonraki ayın gerçek değeri test verisinin içindedir. Böylece 1 adım ileri tahmin 14 kez kontrol edilebilir.
- **Uzun vade (12 ay sonrası):** 12 ay sonrasını test edebilmek için başlangıç noktasının en az 12 ay sonrasında hâlâ gerçek veri bulunmalıdır. Başlangıç noktası test verisinin ortasına ya da sonuna geldiğinde 12 ay sonrası veri setinin dışına, yani bilinmeyen geleceğe taşar; karşılaştırılacak gerçek değer kalmadığı için o noktalarda hata hesaplanamaz.

**Tanım:** Test için ayrılan gözlem sayısı $k$ ve tahmin ufku $h$ olmak üzere, $h$ adım ileri tahmin için hesaplamaya giren gözlem sayısı:

$$N_h = k - h + 1$$

> **Simge notu:** $`N_h`$ *(N h)*: h adım ileri tahminde hesaplamaya giren tahmin sayısı; alt indis hangi ufka ait olduğunu gösterir · $`k`$: saklanan test ayı sayısı · $`h`$: tahmin ufku

Formülün mantığı basit bir saymadır: $h$ adım ileri tahminle kontrol edilebilen ilk test ayı $h$. aydır (eğitim sonundan $h$ adım sonrası), sonuncusu ise $k$. aydır. $h$'den $k$'ye kadar (ikisi de dahil) $k - h + 1$ tane ay vardır; örneğin $k = 14$ ve $h = 12$ için 12., 13. ve 14. aylar, yani $14 - 12 + 1 = 3$ ay. $k = 14$ için: $N_1 = 14$, $N_3 = 12$, $N_6 = 9$, $N_{12} = 3$. Bu değerler, 14.5'teki Weka çıktısının `N` satırıyla birebir aynıdır.

![Ufuk arttıkça N değerinin azalması](../images/ch14_ufuk_N.svg)

*Şekil 14.2 — Saklanan 14 aylık test verisinde ufuk (h) uzadıkça değerlendirilebilen tahmin sayısı azalır. Yeşil hücreler, h adım önceki bir başlangıç noktasından tahmin edilip gerçek değerle karşılaştırılabilen ayları gösterir; mor ok, eğitim sonundan başlayan ilk h adımlık tahmini temsil eder.*

**Yorum:** `N` ne kadar büyükse, hesaplanan hata ortalaması (MAE, RMSE) o kadar güvenilirdir. `N`'nin çok küçüldüğü uzun ufuklarda (ör. N = 3) ortalama hata yalnızca birkaç denemeye dayanır; bu tek bir şanslı ya da şanssız aydan kolayca etkilenebilir. 14.5'teki tabloda 12 adım ileri MAE'nin (17.4) 1 adım ileri MAE'den (31.9) düşük çıkması bunun tipik bir örneğidir: 17.4, yalnızca Ekim–Aralık 1960'a ait üç tahminin ortalamasıdır ve bu aylar zirve aylarından uzak, tahmini görece kolay aylardır. Tablonun sağ tarafındaki uzun vadeli hata değerlerini yorumlarken bu kısıtı göz önünde bulundurun.

**Not —** Held out alanına `12` yazıp 12 adım ileri tahmin isterseniz $N_{12} = 12 - 12 + 1 = 1$ olur; 12 aylık ufkun hatası tek bir tahmine dayanır. Uzun ufukların hatasını güvenilir biçimde ölçmek istiyorsanız test için ayırdığınız dönemi ufuktan belirgin biçimde uzun tutmalısınız (ör. 12 aylık ufuk için 24 ay). Birden fazla başlangıç noktasıyla sistematik değerlendirme fikrinin daha genel hâli Bölüm 16'da (TimeSeriesSplit) ele alınacaktır.

> **Hemen deneyin 14.6 · Test dönemini 24 aya çıkarın: N satırını önceden hesaplayın** (Weka Explorer, ~4 dk)
>
> Önce kâğıt üzerinde: test için $k = 24$ ay ayrılırsa $N_h = k - h + 1$ formülüne göre $N_1$, $N_6$ ve $N_{12}$ kaç olur? Sonra 14.2.1'deki ayarlarla (`LinearRegression`, ufuk `12`) `Evaluation` sekmesinde `Evaluate on held out training` kutusuna **`24`** yazıp `Start` deyin ve `N` ile `Mean absolute error` satırlarını okuyun.
>
> **Ne göreceksiniz:** `N` satırı 24, 23, 22, …, 13 diye azalır; hesabınız $N_1 = 24$, $N_6 = 19$, $N_{12} = 13$ olmalıdır. MAE satırı: 20.3865, 26.3764, 26.7286, 27.5164, 27.3406, 28.6726, 28.6108, 29.0114, 29.5376, 30.8038, 30.9421, 32.1747. Artık her sütun en az 13 tahmine dayanır ve hata ufukla birlikte neredeyse düzenli olarak artar; held out `0.1` (14 ay) ile görülen inişli çıkışlı görünüm kaybolur.
>
> **Şimdi siz:** Aynı ayarlarla (`held out` = `24`) `Base learner`'ı `RandomForest` yapın. Notun 14.5'teki tablosunda görülen "10–12. adımlarda hatanın düşmesi" yine görülür mü? Hatalar neden yukarıdaki tablodan çok daha büyük çıkar?
>
> <details><summary>Cevap</summary>
>
> MAE 53.358'den (1 adım) 72.227'ye (12 adım) çıkar; 8–9. adımdaki küçük dalgalanma (64.139, 62.5787) dışında artış düzenlidir ve uzun ufuklardaki belirgin düşüş görülmez. Hatalar büyüktür, çünkü 24 ay ayrılınca eğitim 1958 Aralık'ta biter ve eğitimdeki en büyük değer 505'tir; ağaç tabanlı model 1959–1960'ın daha yüksek düzeylerini tahmin edemez (Bölüm 13.3). `N` küçükken görülen "uzak ufukta hata düşüyor" izleniminin büyük ölçüde az sayıda tahmine dayanan ortalamadan kaynaklandığı böylece görülür.
>
> </details>

---

[İçindekiler](../Course_notes.md) · [← 13. XGBoost ile Zaman Serisi Tahmini](13_xgboost_ile_zaman_serisi_tahmini.md) · [15. Derin Öğrenme ile Tahmin: LSTM, GRU ve 1D-CNN →](15_derin_ogrenme_ile_tahmin_lstm_gru_ve_1d_cnn.md)
