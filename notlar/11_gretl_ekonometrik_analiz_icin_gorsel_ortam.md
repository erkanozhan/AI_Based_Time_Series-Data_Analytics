[İçindekiler](../Course_notes.md) · [← 10. VAR: Çok Değişkenli Zaman Serisi Modeli](10_var_cok_degiskenli_zaman_serisi_modeli.md) · [12. Yapay Zeka ile Zaman Serisi Analizine Giriş →](12_yapay_zeka_ile_zaman_serisi_analizine_giris.md)

## 11. Gretl: Ekonometrik Analiz için Görsel Ortam

Python ve R'da komut yazarak çalışmak oldukça esnektir, ancak ilk adımda yorucu olabilir. **Gretl** (*GNU Regression, Econometrics and Time-series Library*), özellikle ekonometrik modeller ve zaman serileri için tasarlanmış, **ücretsiz**, açık kaynak ve **grafik arayüzlü** bir programdır:

*   Menüler üzerinden birkaç tıklamayla regresyon, ARIMA, VAR gibi modeller kurulabilir.
*   Aynı işlemler Gretl'in kendi komut diliyle (*hansl*) betik (script) olarak da yazılıp tekrarlanabilir.
*   Zaman serisi yapısını tanımlamayı, otokorelasyonları görmeyi ve durağanlık testlerini yapmayı kolaylaştırır.

Gretl'in menülerine kısa bir giriş için bkz. [Ek B: Gretl'e Giriş](../Gretl.md).

Gretl'i, kod yazmadan "ekonometrik çekirdek modelleri" denemek için pratik bir masaüstü laboratuvarı olarak düşünebilirsiniz. Bu bölümde önceki bölümlerde teorisini gördüğümüz yöntemleri Gretl'de uygulayacağız: ARIMA (Bölüm 7), ADF/KPSS testleri (Bölüm 3.2 ve 7.5), hata metrikleri (Bölüm 8) ve VAR (Bölüm 10). Bu bölümdeki betikler Gretl 2024d sürümüyle çalıştırılıp çıktıları kontrol edilmiştir.

---

### 11.1. Arayüz ve Temel Kavramlar

Gretl'i açtığınızda karşınıza şu bölümler çıkar:

*   **Menü çubuğu:** **File, Tools, Data, View, Add, Sample, Variable, Model, Help**. Zaman serisi çalışmalarında en çok **Data** (veri yapısı), **Add** (log, fark gibi yeni değişkenler), **Variable** (seçili değişken için testler) ve **Model** menüleri kullanılır.
*   **Ana pencere:** Veri kümesinin adı, frekansı ve örnek aralığı üstte; değişken listesi altta görünür.
*   **Değişken listesi:** Veri yüklendikten sonra değişkenlerin adları, numaraları ve açıklamaları burada listelenir. Bir değişkene çift tıklamak değerlerini, sağ tıklamak kısayol menüsünü (grafik, korelogram, testler) açar. Listede her zaman bulunan `const` (0 numaralı), regresyonlardaki sabit terim için Gretl'in kendi oluşturduğu değişkendir.
*   **Konsol ve betik penceresi:** **Tools → Gretl console** ile komutları tek tek çalıştırabilir, **File → Script files → New script** ile betik yazabilirsiniz.

Gretl veri kümelerini üç yapıya ayırır:

1.  **Kesitsel veri** (*cross-section*)
2.  **Zaman serisi** (*time series*)
3.  **Panel veri** (zaman serisi + kesit)

Bir veri kümesini zaman serisi olarak kullanmak için frekansını (aylık, çeyreklik, yıllık vb.) ve başlangıç tarihini bir kez tanımlamak yeterlidir; Gretl bu yapıyı sonraki tüm grafik ve modellerde otomatik kullanır. Tipik bir çalışmanın adımları Şekil 11.1'de özetlenmiştir.

![Gretl'de tipik zaman serisi iş akışı](../images/ch11_gretl_akisi.svg)

*Şekil 11.1 — Gretl'de tipik iş akışı: veri yükle → zaman serisi olarak tanımla → grafik ve keşif → model kur → test ve tahmin. Her adımın menü yolu ve eşdeğer betik komutu gösterilmiştir.*

**Not —** Menü adları sürümden sürüme küçük farklılıklar gösterebilir. Bu bölümde güncel sürümlerdeki adlar kullanılmıştır; eski sürümlerde ARIMA ve VAR, **Model → Time series** altında yer alır.

> **Hemen deneyin 11.1 · AirPassengers Gretl'in içinde zaten var** (Gretl menüleri ve konsol, ~3 dk)
>
> Gretl'i açın ve **File → Open data → Sample file…** seçin. Açılan pencerenin **Gretl** sekmesinde `bjg` satırını ("Box and Jenkins Series G (airline passengers)") bulup çift tıklayın. Ana pencerenin üstündeki veri bilgisine ve değişken listesine bakın. Ardından **Tools → Gretl console** ile konsolu açın ve aşağıdaki satırı yazıp Enter'a basın:
>
> ```gretl
> labels          # değişkenlerin açıklamalarını yazar
> ```
>
> **Ne göreceksiniz:** Veri aylıktır ve 1949:01–1960:12 aralığını kapsar (144 gözlem). Listede `const` dışında iki değişken vardır: `g` ve `lg`. `labels` çıktısı `g: international airline passengers` ve `lg: natural logarithm of g` yazar. `bjg`, AirPassengers'ın Gretl ile birlikte gelen kopyasıdır (özet istatistiklerini Hemen deneyin 1.5'te almıştık; ortalama 280.299, en küçük 104, en büyük 622) ve zaman serisi yapısı (aylık, 1949:01'den başlayan) önceden tanımlanmıştır. Bu yüzden 11.2.2'deki adıma gerek kalmaz.
>
> **Şimdi siz:** Aynı pencerenin **Gretl** sekmesinden `nile` dosyasını açın (Gretl açık olan veriyi kapatmak isteyip istemediğinizi sorarsa **Yes** deyin). Verinin frekansı, tarih aralığı ve gözlem sayısı nedir? Konsolda `labels` yazarak değişkenin ne ölçtüğünü öğrenin.
>
> <details><summary>Cevap</summary>
>
> Yıllık (*annual*) veri: 1871–1970, 100 gözlem. Tek değişken `nile`'dır; açıklaması `Annual flow of the Nile at Aswan, 10^8 cubic meters`, yani Nil nehrinin Asvan'daki yıllık akış miktarı (10⁸ m³ cinsinden). Bu veriyi 11.4'teki alıştırmada kullanacağız.
>
> </details>

---

### 11.2. Veri Hazırlığı

#### 11.2.1. Veri Yükleme

AirPassengers gibi bir CSV dosyasını Gretl'e aktarmak için:

1.  **File → Open data → User file…** seçilir.
2.  Dosya türü olarak CSV (ya da "all files") seçilip `AirPassengers.csv` işaretlenir.
3.  Sütun ayırıcı (virgül, noktalı virgül) genellikle otomatik algılanır; gerekirse elle seçilir.
4.  Sütun adlarının doğru okunduğu kontrol edilir (ör. `Month`, `Passengers`).

Gretl ayrıca Excel, Stata, SPSS ve kendi biçimi olan `.gdt` dosyalarını da aynı menüden açabilir. Gretl'in tarihleri kendiliğinden tanıması için iki koşul gerekir: ilk sütunun başlığı boş ya da `obs` veya `date` gibi bir gözlem etiketi adı olmalı ve tarihler `1949-01`, `1949:01`, `1949-01-01` gibi tanınan bir biçimde yazılmalıdır. Bölüm 10'daki `macro.csv` dosyası (`date` başlığı, `2015-01-01` biçimi) bu yüzden doğrudan aylık seri olarak açılır. `AirPassengers.csv`'de ise ilk sütunun başlığı `Month` olduğu için Gretl bu sütunu sıradan bir metin değişkeni sayar ve veriyi tarihsiz (*undated*) bir tablo olarak açar; bu durumda bir sonraki adım gerekir.

> **Hemen deneyin 11.2.1 · Kendi CSV dosyanızı yazın: ilk sütunun başlığı neden önemli?** (Not Defteri ve Gretl menüleri, ~5 dk)
>
> Not Defteri'ni açın ve aşağıdaki yedi satırı yazın (altı aylık uydurma bir satış verisi):
>
> ```text
> date,satis
> 2024-01,120
> 2024-02,135
> 2024-03,128
> 2024-04,150
> 2024-05,162
> 2024-06,171
> ```
>
> Dosyayı **Farklı Kaydet** ile `satis.csv` adıyla kaydedin (Kayıt türü: **Tüm dosyalar**; yoksa dosya `satis.csv.txt` olur). Gretl'de **File → Open data → User file…** ile bu dosyayı açın.
>
> **Ne göreceksiniz:** Ana pencerenin üstünde verinin aylık olduğu ve 2024:01–2024:06 aralığını kapsadığı yazar; değişken listesinde (`const` dışında) yalnızca `satis` vardır. İlk sütunun başlığı `date` olduğu ve tarihler `2024-01` biçiminde yazıldığı için Gretl bu sütunu değişken değil **gözlem etiketi** sayar ve zaman yapısını kendisi kurar.
>
> **Şimdi siz:** Not Defteri'nde yalnızca ilk satırdaki `date` sözcüğünü `Ay` yapıp kaydedin ve dosyayı Gretl'de yeniden açın (Gretl veri yapısını tanımlamak isteyip istemediğinizi sorarsa şimdilik **No** deyin). Neyin değişeceğini önce tahmin edin.
>
> <details><summary>Cevap</summary>
>
> Bu kez değişken listesinde `Ay` ve `satis` adlı iki değişken görünür ve veri tarihsiz (*undated*) açılır: gözlemler 1'den 6'ya kadar numaralanır. Gretl `Ay` sütununu metin içerikli sıradan bir değişken olarak okumuştur. Bu, notta anlatılan `AirPassengers.csv` dosyasındaki `Month` başlığının yarattığı durumun aynısıdır; çözümü bir sonraki alt başlıktaki **Data → Dataset structure…** adımıdır (konsol karşılığı: `setobs 12 2024:01 --time-series`).
>
> </details>

#### 11.2.2. Zaman Serisi Olarak Tanımlama

1.  **Data → Dataset structure…** seçilir.
2.  Açılan sihirbazda sırasıyla:
    *   **Time series** seçilir,
    *   **Frequency** olarak *monthly* seçilir,
    *   **Starting observation** olarak `1949:01` girilir.
3.  Onaylandığında Gretl her satırı bir aya karşılık gelen bir gözlem olarak kabul eder; ana pencerenin üstünde "Monthly data, 1949:01–1960:12" gibi bir bilgi görünür.

Bu aşamadan sonra grafiklerde tarih ekseni doğru görünür, mevsimsel fark gibi işlemler ve ARIMA modelleri ek bir ayar gerektirmeden çalışır. Gretl'de tarihler `yıl:dönem` biçiminde yazılır: `1949:01` 1949'un 1. ayı, çeyreklik veride `2015:3` 2015'in 3. çeyreğidir.

> **Hemen deneyin 11.2.2 · Yanlış frekans seçerseniz ne olur?** (Gretl menüleri, ~3 dk)
>
> `data/AirPassengers.csv` dosyasını **File → Open data → User file…** ile açın (tarihsiz açılır). Şimdi bilerek yanlış yapın: **Data → Dataset structure…** sihirbazında **Time series** seçin, **Frequency** olarak *quarterly* (çeyreklik) seçin ve **Starting observation** kutusuna `1949:1` yazın. Onayladıktan sonra ana pencerenin üstündeki tarih aralığına bakın.
>
> **Ne göreceksiniz:** Aralık **1949:1 – 1984:4** olarak görünür. Gretl hata vermez: 144 satırı 144 çeyrek, yani 36 yıl sayar. Frekansın doğru olup olmadığını Gretl bilemez; bunu kontrol etmek sizin işinizdir. Yanlış frekansla mevsimsel fark (`sdiff`) 12 ay yerine 4 dönem geriye bakar ve mevsimsel ARIMA yanlış periyotla ($s = 4$) kurulur.
>
> **Şimdi siz:** (1) Aynı sihirbazda *annual* (yıllık) seçip başlangıca `1949` yazarsanız son gözlem hangi yıl olur? Önce kafadan hesaplayın. (2) Ardından *monthly* ve `1949:01` ile yapıyı düzeltin.
>
> <details><summary>Cevap</summary>
>
> (1) 144 gözlem yıllık sayılırsa son yıl 1949 + 143 = **2092** olur; aralık 1949–2092 görünür. (2) Aylık ve `1949:01` seçince aralık doğru biçimde **1949:01 – 1960:12** olur. Konsolda aynı denemeler sırasıyla `setobs 4 1949:1 --time-series`, `setobs 1 1949 --time-series` ve `setobs 12 1949:01 --time-series` komutlarıyla yapılabilir.
>
> </details>

---

### 11.3. Keşifsel Analiz ve Modelleme

#### 11.3.1. Grafikler, Özet İstatistikler ve Dönüşümler

*   **View → Graph specified vars → Time series plot…** ile seçilen değişkenlerin zaman grafiği çizilir (ya da değişkene sağ tıklayıp **Time series plot**).
*   **View → Summary statistics** ile ortalama, standart sapma, minimum, maksimum gibi özet istatistikler görülür. `Passengers` için ortalama yaklaşık 280, en küçük değer 104, en büyük değer 622'dir (bin yolcu).
*   **Add** menüsünden yeni değişkenler türetilir: **Add → Logs of selected variables** (log dönüşümü), **Add → First differences of selected variables** (birinci fark), **Add → Seasonal differences of selected variables** (mevsimsel fark: bu ayın değeri eksi bir yıl önceki aynı ayın değeri). Gretl bunları `l_Passengers`, `d_Passengers`, `sd_Passengers` gibi adlarla listeye ekler.

`Passengers` değişkeninin grafiğinde AirPassengers'ın bilinen yapısı görülür: artan trend, her yıl tekrarlayan mevsimsellik ve zamanla büyüyen dalgalanmalar (Bölüm 2.5).

> **Hemen deneyin 11.3.1 · Log, fark ve mevsimsel fark: kaç gözlem kaybolur?** (Gretl menüleri, ~4 dk)
>
> 11.1'deki gibi `bjg` örnek dosyasını açın. Değişken listesinde `g`'yi seçin ve sırasıyla **Add → Logs of selected variables**, **Add → First differences of selected variables**, **Add → Seasonal differences of selected variables** seçeneklerine tıklayın (her seferinde `g` seçili olsun). Listeye `l_g`, `d_g` ve `sd_g` eklenir. Sonra **View → Summary statistics** ile özet tabloyu açıp bu üç değişkenin satırlarına bakın.
>
> **Ne göreceksiniz:** `Missing obs.` (eksik gözlem) sütununda `l_g` için 0, `d_g` için **1**, `sd_g` için **12** yazar: birinci fark ilk ayda, mevsimsel fark ilk yılın 12 ayında hesaplanamaz, çünkü karşılaştırılacak önceki değer yoktur. Ortalamalar: `d_g` 2.2378 (bir aydan diğerine ortalama yaklaşık 2 bin yolcu artış), `sd_g` 31.773 (geçen yılın aynı ayına göre ortalama yaklaşık 32 bin yolcu artış). `sd_g`'nin en küçük değeri −8, en büyüğü 74'tür.
>
> **Şimdi siz:** `sd_g`'ye çift tıklayıp değerlerini inceleyin. Hangi aylarda `sd_g` negatiftir, yani yolcu sayısı bir yıl önceki aynı ayın altına düşmüştür? `d_g`'nin en küçük değeri (−101) hangi aydadır ve neden bu kadar büyük bir düşüş olur?
>
> <details><summary>Cevap</summary>
>
> `sd_g` yalnızca üç ayda negatiftir: **1954:02 (−8), 1954:03 (−1), 1954:04 (−8)**; 1954 baharında yolcu sayısı 1953'ün aynı aylarının biraz altında kalmıştır. `d_g = −101` **1958:09**'dadır (Ağustos 505 → Eylül 404). Bu bir kriz değil, yaz zirvesinin ardından gelen olağan mevsimsel düşüştür: birinci fark trendi temizler ama mevsimselliği temizlemez; mevsimsel fark (`sd_g`) ise bu tür ay etkilerini ortadan kaldırır.
>
> </details>

#### 11.3.2. Basit Doğrusal Regresyon ve Artıkların İncelenmesi

Gretl'in güçlü yanlarından biri, regresyonun birkaç tıklamayla kurulabilmesidir:

1.  **Model → Ordinary Least Squares…** (sıradan en küçük kareler, OLS) seçilir.
2.  **Dependent variable** (bağımlı değişken, yani açıklanmak istenen) olarak ör. `Passengers` seçilir.
3.  **Regressors** (açıklayıcı değişkenler) olarak zaman trendi, mevsimsel kuklalar ya da gecikmeler eklenir. Bunlar önceden **Add → Time trend**, **Add → Periodic dummies** ve **Add → Lags of selected variables** ile oluşturulabilir. Mevsimsel **kukla** (*dummy*) değişken, ilgili ayda 1, diğer aylarda 0 değerini alan bir değişkendir (ör. her Temmuz'da 1).
4.  **OK** dendiğinde katsayı tahminleri, t-istatistikleri (katsayının standart hatasına oranı; mutlak değeri yaklaşık 2'den büyükse katsayı genellikle anlamlıdır), R-kare (açıklanan varyans oranı, Bölüm 10.3.3) ve bilgi kriterleri ayrı bir model penceresinde gösterilir.

Model penceresinde **Graphs → Residual plot → Against time** ile artıkların zaman grafiği, **Tests → Autocorrelation** ile artıklarda otokorelasyon olup olmadığı incelenir. Zaman serisi regresyonlarında artıklar genellikle güçlü otokorelasyon gösterir; bu, ARIMA gibi dinamik modellere geçme gereğine işaret eder.

> **Hemen deneyin 11.3.2 · Trend ve mevsim kuklaları R-kareyi ne kadar artırır?** (Gretl menüleri, ~5 dk)
>
> `bjg` açıkken **Add → Time trend** (listeye `time` eklenir: 1, 2, 3, …) ve **Add → Periodic dummies** (listeye `dm1` … `dm12` eklenir: `dm1` her Ocak'ta 1, diğer aylarda 0) seçin. Sonra:
>
> 1.  **Model → Ordinary Least Squares…** açın; **Dependent variable** olarak `g`, **Regressors** olarak yalnızca `time` seçin (`const` zaten listededir) ve **OK** deyin. `R-squared` satırını not edin.
> 2.  Aynı modeli yeniden açıp regresörlere `dm2` … `dm12`'yi de ekleyin. `dm1`'i eklemeyin: 12 kuklanın toplamı her ay 1 ettiği için `const` ile birlikte tam doğrusal bağımlılık oluşur (Gretl eklerseniz birini kendisi atar).
>
> **Ne göreceksiniz:** Yalnızca trendle **R-squared = 0.853638** ve `Durbin-Watson = 0.537194` (Durbin-Watson artıklarda birinci derece otokorelasyonu ölçer; 2'ye yakınsa otokorelasyon yok, 0'a yakınsa güçlü pozitif otokorelasyon var, bkz. Bölüm 10.9); trend + 11 kuklayla **R-squared ≈ 0.9559**. Kuklalar, trendin açıklayamadığı değişimin büyük kısmını (mevsim etkisini) açıklar.
>
> **Şimdi siz:** Aynı iki modeli bağımlı değişken `l_g` ile kurun (`l_g` yoksa önce **Add → Logs of selected variables**). R-kare ne olur? Kuklalı log modelinin penceresinde **Tests → Autocorrelation** (gecikme 12) çalıştırın: Ljung-Box p-değeri kaçtır?
>
> <details><summary>Cevap</summary>
>
> `l_g ~ time`: R-kare 0.9015; `l_g ~ time + dm2…dm12`: **R-kare 0.983468**. Log ölçeğinde mevsim etkisi her yıl aynı oranda olduğu için sabit kuklalar daha iyi uyar (Bölüm 2.4'teki çarpımsal yapı). Örneğin `dm7` katsayısı 0.300619'dur: Temmuz, Ocak'a göre yaklaşık $e^{0.30} \approx 1.35$ kat, yani yaklaşık %35 daha kalabalıktır. Ancak `Durbin-Watson` 0.425184'tür ve otokorelasyon testinde `Ljung-Box Q' = 393.56`, p-değeri `8.74e-77`'dir: R-kare çok yüksek olsa da artıklar güçlü otokorelasyonludur. Bu, yukarıdaki paragrafta söylendiği gibi dinamik (ARIMA) bir modele geçme gereğini gösterir.
>
> </details>

#### 11.3.3. ARIMA Modelleri

Model kurmadan önce serinin durağanlığını sınamak gerekir (Box-Jenkins adım 0, Bölüm 7.4); Gretl'de bunun nasıl yapıldığını 11.4'te göreceğiz. Burada log ve farkı alınmış serinin durağan olduğunu biliyoruz (Bölüm 7.6).

ARIMA ve SARIMA'nın teorisi Bölüm 7'de anlatılmıştır; burada yalnızca Gretl'deki kurulumu ele alıyoruz:

1.  **Model → Univariate time series → ARIMA…** seçilir.
2.  **Dependent variable** olarak ör. `l_Passengers` (log alınmış seri) seçilir.
3.  Model dereceleri girilir:
    *   **AR order** ($p$), **Difference** ($d$), **MA order** ($q$),
    *   Mevsimsel kısım için **Seasonal AR** ($P$), **Seasonal difference** ($D$), **Seasonal MA** ($Q$); periyot ($s = 12$) veri frekansından otomatik alınır.
    *   Gretl varsayılan olarak modele bir sabit terim ekler. $d = 1$ ve $D = 1$ olan bir modelde sabit gereksizdir (iki kez fark alınmış seride sabit, giderek hızlanan bir trend anlamına gelir); R'daki `Arima()` ile aynı sonucu almak için **Include a constant** kutusunun işaretini kaldırın.
4.  **OK** dendiğinde parametre tahminleri, standart hatalar, log-olabilirlik ve bilgi kriterleri (AIC, BIC, HQ) listelenir.
5.  Model penceresinden **Graphs → Residual correlogram** ile artıkların ACF/PACF grafikleri, **Analysis → Forecasts…** ile tahminler ve güven aralıkları elde edilir.

> **Hemen deneyin 11.3.3 · Sabit kutusunu işaretleyip kaldırın: AIC ne kadar değişir?** (Gretl menüleri, ~4 dk)
>
> `bjg` açık ve `l_g` listede olsun (yoksa **Add → Logs of selected variables**). **Model → Univariate time series → ARIMA…** açın; **Dependent variable** olarak `l_g` seçin; **AR order** 0, **Difference** 1, **MA order** 1; **Seasonal AR** 0, **Seasonal difference** 1, **Seasonal MA** 1 girin. **Include a constant** kutusunu bu kez **işaretli bırakın** ve **OK** deyin. Sonra aynı pencereyi açıp kutunun işaretini kaldırarak modeli yeniden kurun.
>
> **Ne göreceksiniz:** Sabitli modelde `const = −0.000162569`, p-değeri **0.8675**'tir (anlamsız) ve `Akaike criterion` **−481.4207**'dir. Sabitsiz modelde AIC **−483.3930**, `theta_1 = −0.401823`, `Theta_1 = −0.556937`'dir (11.5'teki airline model). Log-olabilirlikler neredeyse aynıdır (244.7104 ve 244.6965); AIC her ek parametre için 2 birim ceza eklediğinden (Bölüm 7.4) işe yaramayan sabit, AIC'yi yaklaşık 2 birim kötüleştirir.
>
> **Şimdi siz:** Sabitsiz modelde normal kısmı **AR order** 1, **MA order** 0 yaparak ARIMA(1,1,0)(0,1,1) kurun. AIC'si airline modelden daha mı iyi (daha küçük mü) olur? Önce tahmin edin.
>
> <details><summary>Cevap</summary>
>
> AIC **−481.48** çıkar (11.5'teki betik çıktısındaki değerle aynı). Bu, airline modelin −483.39'undan büyüktür; yani airline model daha iyidir. İki model aynı fark serisini ($d = 1$, $D = 1$) modellediği için AIC'leri karşılaştırılabilir.
>
> </details>

---

### 11.4. Model Doğrulama: Otokorelasyon ve Durağanlık Testleri

Serinin durağan olup olmadığı ve artıkların otokorelasyon içerip içermediği, zaman serisi modellemesinin temel kontrolleridir (Bölüm 3.2, 7.5 ve 7.6.6). Gretl'de:

| Amaç | Menü yolu | Komut |
| --- | --- | --- |
| Değişkenin ACF/PACF grafiği | Değişkeni seç → **Variable → Correlogram** | `corrgm x 36` |
| ADF birim kök testi | **Variable → Unit root tests → Augmented Dickey-Fuller test** | `adf 12 x --c --ct` |
| KPSS durağanlık testi | **Variable → Unit root tests → KPSS test** | `kpss 12 x` |
| Artıklarda otokorelasyon (ARIMA'da Ljung-Box, OLS'de Breusch-Godfrey, VAR'da Rao F) | Model penceresi → **Tests → Autocorrelation** | `modtest --autocorr` |
| Artıkların normalliği | Model penceresi → **Tests → Normality of residual** | `modtest --normality` |

Komutlardaki sayılar ve seçenekler şöyle okunur: `corrgm x 36` `x` değişkeninin ilk 36 gecikmedeki ACF ve PACF değerlerini tablo olarak yazar (ve grafiğini çizer). `adf 12 x --c --ct` ADF testini 12 gecikmeli farkla yapar; `--c` yalnızca sabitli, `--ct` sabit ve trendli sürümü ister, ikisi birlikte yazılınca iki sonuç da raporlanır. `kpss 12 x` KPSS testini 12 gecikmelik bant genişliğiyle yapar. `modtest` her zaman **en son tahmin edilen modele** uygulanır; bu yüzden bir modeli kurduktan hemen sonra yazılır.

ADF testinde $H_0$ "birim kök vardır (seri durağan değildir)", KPSS testinde ise $H_0$ "seri durağandır" şeklindedir; iki testi birlikte kullanmak daha güvenilir bir karar verir. Ljung-Box testinde $H_0$ "artıklarda otokorelasyon yoktur" hipotezidir ve iyi bir modelde reddedilmemesi (p > 0.05) istenir. Normallik testinde de $H_0$ "artıklar normal dağılır" şeklindedir (H₀/H₁ ve p-değeri: bkz. Bölüm 3.2).

Bu testler, ARIMA kurarken ya da daha sonra LSTM/GRU gibi modellere (Bölüm 15) geçmeden önce serinin yapısını anlamak için de yararlıdır.

> **Hemen deneyin 11.4 · Nil nehri: ADF ve KPSS aynı şeyi mi söylüyor?** (Gretl konsolu, ~5 dk)
>
> **File → Open data → Sample file…** → **Gretl** sekmesi → `nile` dosyasını açın (yıllık, 1871–1970). **Tools → Gretl console** açıp şu satırları yazın:
>
> ```gretl
> corrgm nile 10        # ilk 10 gecikmenin ACF/PACF tablosu (grafik de açılır)
> adf 4 nile --c        # sabitli ADF testi, 4 gecikmeli farkla
> kpss 4 nile           # KPSS testi, H0: seri durağandır
> ```
>
> Yıllık ve 100 gözlemlik bir seride 12 yıllık gecikme çok fazla olacağı için tablodaki `12` yerine `4` kullandık. Menüyle yapmak isterseniz `nile` seçiliyken **Variable → Unit root tests → Augmented Dickey-Fuller test** ve **KPSS test** pencerelerinde gecikme sayısını 4 yapın (pencerede gecikmeyi otomatik küçülten bir seçenek işaretliyse kaldırın; aksi hâlde sayılar biraz farklı çıkabilir).
>
> **Ne göreceksiniz:** ACF 1. gecikmede 0.4984'tür ve 8. gecikmeye kadar anlamlı kalır. ADF'de `tau_c(1) = -2.78196`, `asymptotic p-value 0.06085`: p > 0.05 olduğu için "birim kök vardır" hipotezi %5 düzeyinde **reddedilemez** (ama sınırdadır). KPSS'de `Test statistic = 0.965435`, %1 kritik değer 0.734'ün üstündedir ve `P-value < .01`: "seri durağandır" hipotezi **reddedilir**. İki test birlikte okununca düzey seri durağan görünmez.
>
> **Şimdi siz:** Konsolda `diff nile` yazıp farkı alınmış `d_nile` serisi için `adf 4 d_nile --c` ve `kpss 4 d_nile` çalıştırın. Sonuçları önce tahmin edin.
>
> <details><summary>Cevap</summary>
>
> ADF: `tau_c(1) = -6.62375`, p-değeri `3.287e-09`, birim kök hipotezi açıkça reddedilir. KPSS: `Test statistic = 0.0293573`, `P-value > .10`, durağanlık hipotezi reddedilmez. İki test bu kez aynı yönü gösterir: 7.5a'da R ile bulduğumuz gibi, birinci farkı alınmış Nil serisi durağandır ($d = 1$ yeterlidir).
>
> </details>

---

### 11.5. Komut Dili ile Otomasyon: ARIMA Betiği

Menülerle yapılan her işlem Gretl'in komut dilinde de yazılabilir. Betik kullanmanın avantajı, analizin **tekrarlanabilir** olmasıdır: veri güncellendiğinde aynı adımlar tek tuşla yeniden çalıştırılır. Aşağıdaki betik, AirPassengers üzerinde Bölüm 7'deki Box-Jenkins adımlarını (dönüşüm → durağanlık → korelogram → model seçimi → artık kontrolü → tahmin) ve Bölüm 8'deki eğitim-test değerlendirmesini uygular.

Betiği çalıştırmak için **File → Script files → New script** ile açılan pencereye yapıştırıp **Run** (dişli simgesi) düğmesine basabilir ya da `.inp` dosyası olarak kaydedip komut satırından `gretlcli -b betik.inp` ile çalıştırabilirsiniz (`gretlcli` Gretl'in komut satırı sürümüdür, `-b` "batch", yani betiği baştan sona soru sormadan çalıştır demektir).

Tam betik yaklaşık 160 satır olduğu için ayrı bir dosyaya taşındı. Aşağıda her adımın kilit komutları yer alıyor; adım numaraları (1–8), dosyadaki bölüm başlıklarıyla ve aşağıdaki "Betiğin çıktıları nasıl okunur?" listesiyle aynıdır.

```gretl
# ARIMA betiğinin kilit satırları (tam betik: Codes/gretl/ch11_arima_airpassengers.inp)
open "data/AirPassengers.csv"                    # 1) veri (depo kök dizininden)
setobs 12 1949:01 --time-series
rename Passengers passengers

series lnpass = log(passengers)                  # 2) log dönüşümü
series ddlnpass = diff(sdiff(lnpass))            # 3) birinci + mevsimsel fark
adf 12 lnpass --c --ct
adf 12 ddlnpass --c
kpss 12 ddlnpass

corrgm ddlnpass 36                               # 4) korelogram

arima 0 1 1 ; 0 1 1 ; lnpass --nc --quiet        # 5) aday modeller (burada airline model)
scalar aic_airline = $aic

arima 0 1 1 ; 0 1 1 ; lnpass --nc                # 6) seçilen model ve artık testleri
series uhat = $uhat
modtest --autocorr

smpl 1949:01 1959:12                             # 7) eğitim: 1949-1959, test: 1960
arima 0 1 1 ; 0 1 1 ; lnpass --nc --quiet
smpl full
fcast 1960:01 1960:12 lnpass_test --dynamic

arima 0 1 1 ; 0 1 1 ; lnpass --nc --quiet        # 8) tüm veriyle 1961 tahmini
dataset addobs 12
fcast 1961:01 1961:12 --dynamic                  #    tabloyu yazdır
fcast 1961:01 1961:12 lnpass_f --dynamic         #    seriye kaydet
series pass_f = exp(lnpass_f)                    #    log ölçeğinden yolcu sayısına dön
```

Komutların ne yaptığı:

*   `open "dosya"` veri dosyasını açar. `setobs 12 1949:01 --time-series` veriyi zaman serisi olarak tanımlar: `12` frekanstır (yılda 12 gözlem, yani aylık), `1949:01` ilk gözlemin tarihidir. Menüdeki **Data → Dataset structure** adımının (11.2.2) komut karşılığıdır. `rename eski yeni` değişkenin adını değiştirir.
*   `series yeni = ifade` yeni bir seri (sütun) oluşturur. `log()` doğal logaritma, `exp()` onun tersi (log ölçeğinden geri dönüş), `diff(x)` birinci fark ($x_t - x_{t-1}$), `sdiff(x)` mevsimsel fark ($x_t - x_{t-12}$) alır; `diff(sdiff(lnpass))` ikisini art arda uygular.
*   `arima p d q ; P D Q ; seri` modeli kurar: noktalı virgüller normal kısmı, mevsimsel kısmı ve bağımlı değişkeni ayırır. `--nc` (*no constant*) sabit terim eklememesini söyler (11.3.3'teki not). `--quiet` ayrıntılı çıktıyı bastırır.
*   `$aic`, `$uhat` gibi `$` ile başlayan adlar **erişimcilerdir** (*accessor*): son kurulan modelden bir sonuç çeker (`$aic` AIC değeri, `$uhat` artıklar). `scalar ad = ...` tek bir sayıyı bir ada kaydeder.
*   `smpl başlangıç bitiş` analizi verinin bir bölümüyle sınırlar; `smpl full` tüm veriye geri döner. Modeli 1959 sonuna kadarki veriyle kurup 1960'ı tahmin etmek, Bölüm 8'deki eğitim-test ayrımının Gretl'deki karşılığıdır.
*   `fcast başlangıç bitiş ad --dynamic` tahmin üretir. `--dynamic` her ay için bir önceki ayın gerçek değerini değil **tahminini** kullanır; yani 1960 değerleri modele hiç gösterilmez ve sonuç gerçek bir 12 aylık ileri tahmindir. Sona bir seri adı yazılırsa tahminler sessizce o seriye kaydedilir; ad yazılmazsa tahmin, standart hata ve %95 aralık tablo olarak ekrana basılır.
*   `dataset addobs 12` veri setinin sonuna 12 boş ay ekler; gelecek tahminlerinin yazılacağı yer böylece açılır.

> **Uygulama dosyası:** [`Codes/gretl/ch11_arima_airpassengers.inp`](../Codes/gretl/ch11_arima_airpassengers.inp)
>
> Betiğin tamamı bu dosyada. Gretl'de **File → Script files → Open user file…** ile açıp **Run** (dişli simgesi) düğmesiyle çalıştırabilir ya da depo kök dizininde `gretlcli -b Codes/gretl/ch11_arima_airpassengers.inp` komutunu kullanabilirsiniz. Betik veriyi `data/AirPassengers.csv` yolundan açar (depo kök dizininden çalıştırıldığı varsayılır); Gretl dosyayı bulamazsa `open` satırına dosyanın tam yolunu yazın. Betik nerede çalıştığını `$sysinfo.gui_mode` ile kendisi anlar: Gretl arayüzünde grafikler ayrı pencerelerde açılır; `gretlcli` ile çalıştırıldığında ise ekran penceresi betiği bekletmesin diye altı grafik `Codes/gretl/` klasörüne PNG dosyası olarak yazılır (`ch11_grafik1_yolcu.png` … `ch11_grafik6_tahmin_1961.png`). Bunu her grafikten önceki `string G = EKRAN ? "display" : "...png"` satırı sağlar; `--output="@G"` bu satırdaki metni kullanır.

**Betiğin çıktıları nasıl okunur?**

*   **Adım 2–4:** Betik seriyi çizip özet istatistiklerini aldıktan sonra, büyüyen dalgalanmalar nedeniyle log alır (`lnpass`); 4. adımdaki 36 gecikmelik korelogramda mevsimsel örüntü (12, 24, 36. gecikmeler) incelenir. Adım 3'te `lnpass` için ADF p-değeri yüksektir (sabitli 0.31, sabit+trendli 0.82: durağan değil), `ddlnpass` için düşüktür (yaklaşık 0.0002: durağan). KPSS'de ise tersi beklenir: `ddlnpass` için test istatistiği 0.103, %5 kritik değerin (0.462) altında kalır ve p > 0.10 olur; "seri durağandır" hipotezi reddedilmez. Bu, Bölüm 7'deki $d = 1$, $D = 1$ seçimini destekler.
*   **Adım 5:** Betik beş aday modeli `--quiet` ile kurar, AIC'lerini `$aic` ile saklar ve `printf` (biçimlendirilmiş yazdırma) ile tablo hâlinde yazdırır; en küçük AIC değerine sahip model seçilir. AIC'ler ancak aynı seri üzerinde kurulan modeller arasında karşılaştırılabilir: $d = 1$, $D = 1$ olan dört mevsimsel model aynı fark serisini modellediği için karşılaştırılabilir. Bunlar arasında airline model en iyisidir (AIC −483.39; diğerleri −481.89, −481.48, −477.40). ARIMA(1,1,1) mevsimsel fark içermez; AIC'si (−242.63) farklı bir seri üzerinden hesaplandığı için diğerleriyle doğrudan karşılaştırılamaz. Bu model ancak mevsimsel yapıyı hiç modellemediği için kötü bir seçimdir ve artıklarında 12. gecikmede güçlü korelasyon kalır.
*   **Adım 6:** Airline modelin katsayıları $\theta_1 \approx -0.40$ ve $\Theta_1 \approx -0.56$'dır (Gretl çıktısında `theta_1` ve `Theta_1`); ikisi de çok anlamlıdır (p < 0.001). Betik artıkları çizer, korelogramlarını alır ve Ljung-Box ile normallik testlerini yapar: `modtest --autocorr` Ljung-Box p-değeri 0.57'dir (0.05'ten büyük) ve artık korelogramında anlamlı çubuk neredeyse kalmaz: model serideki yapıyı yakalamıştır. Normallik testi p-değeri 0.17'dir; artıkların normal olduğu hipotezi reddedilmez.
*   **Adım 7:** 1960 tahminleri `exp()` ile yolcu sayısına çevrildikten sonra RMSE = 18.59, MAE = 13.26 (yolcu sayısı biriminde, bin kişi) ve MAPE = %2.90 çıkar. Bunlar, aynı modelin aynı eğitim-test ayrımıyla R'da elde edilen sonuçlarla (Bölüm 8.3) birebir aynıdır. `--nc` seçeneği kullanılmazsa Gretl'in eklediği sabit terim yüzünden değerler biraz farklı çıkar (18.68, 13.37, %2.93).
*   **Adım 8:** `fcast` tablosu her ay için log ölçeğinde tahmin, standart hata ve %95 aralığı verir. `exp(lnpass_f)` ile yolcu sayısına çevrilen `pass_f` değerleri 1961 Ocak için yaklaşık 450, Temmuz için 670 bin yolcudur; betik bunları gerçek seriyle birlikte çizer ve grafik trendin ve mevsimsel örüntünün 1961'e taşındığını göstermelidir.

> **Simge notu:** $`\theta_1`$ *(teta 1)*: normal MA(1) katsayısı · $`\Theta_1`$ *(büyük teta 1)*: mevsimsel MA(1) katsayısı (Bölüm 7.3)

**Not —** `exp()` ile geri dönüşün ortalamayı değil medyanı verdiğine ilişkin not için bkz. 7.6.7.

> **Hemen deneyin 11.5 · Kısa bir betikle 24 aylık test: ufuk uzayınca hata** (Gretl betiği, ~5 dk)
>
> **File → Script files → New script** ile boş bir betik penceresi açın, aşağıdaki satırları yapıştırıp **Run** (dişli simgesi) düğmesine basın. Betik `bjg` örnek dosyasını kullandığı için dosya yolu sorunu çıkmaz.
>
> ```gretl
> open bjg                                   # Gretl'in AirPassengers kopyası (aylık)
> logs g                                     # l_g = log(g)
> smpl 1949:01 1958:12                       # eğitim: 1949-1958 (120 ay)
> arima 0 1 1 ; 0 1 1 ; l_g --nc --quiet     # airline model
> smpl full
> fcast 1959:01 1960:12 lf --dynamic         # 24 ay ileri dinamik tahmin -> lf
> series gf = exp(lf)                        # log ölçeğinden yolcu sayısına dön
> smpl 1959:01 1960:12                       # yalnızca test dönemi
> print g gf --byobs                         # gerçek ve tahmin yan yana
> printf "MAPE = %.2f\n", 100 * mean(abs((g - gf) / g))
> ```
>
> **Ne göreceksiniz:** 24 satırlık bir tablo ve en altta `MAPE = 8.52`. Tahminler (`gf`) her ayda gerçek değerin altında kalır; örneğin 1960:07'de gerçek 622, tahmin 542.4208'dir. Aynı model 1959 sonuna kadarki veriyle kurulup yalnızca 1960 tahmin edildiğinde (11.5, Adım 7) MAPE %2.90'dı.
>
> **Şimdi siz:** Son satırdan önce `smpl 1959:01 1959:12` ekleyerek yalnızca ilk 12 ayın (1959) MAPE'sini, sonra `smpl 1960:01 1960:12` ile ikinci 12 ayın (1960) MAPE'sini hesaplayın. Hangisi daha büyük çıkar?
>
> <details><summary>Cevap</summary>
>
> 1959: **%6.75**, 1960: **%10.28**. Tahmin ufku uzadıkça (1–12 adım yerine 13–24 adım) hata büyür. Tahminlerin sistematik olarak düşük kalmasının bir nedeni de eğitimin son yılının zayıf olmasıdır: yıllık toplam yolcu 1957'den 1958'e yalnızca %3.4 artmıştır (4421 → 4572), 1959'da ise %12.4 artmıştır (5140). Model 1958'in yavaş büyümesini geleceğe taşır.
>
> </details>

---

### 11.6. Gretl ile VAR Kurulumu

VAR modelinin teorisi (tanım, gecikme seçimi, stabilite, Granger nedenselliği, IRF ve FEVD) Bölüm 10'da anlatılmıştır. Burada aynı analizin Gretl'de nasıl yapıldığını görüyoruz. Örnekte Bölüm 10.9'daki `data/macro.csv` dosyası, yani `inflation`, `interest` ve `exchange` adlı üç aylık seri kullanılmaktadır.

#### 11.6.1. Menü ile VAR

1.  **Gecikme seçimi:** **Model → Multivariate time series → VAR lag selection…** seçilir; değişkenler ve en büyük gecikme (ör. 8) girilir. Gretl her gecikme için AIC, BIC ve HQC değerlerini listeler ve her kriterin en iyi değerini yıldızla işaretler (Bölüm 10.4).
2.  **Modeli kurma:** **Model → Multivariate time series → Vector Autoregression…** seçilir.
    *   **Endogenous variables** (içsel değişkenler, yani modelin birbirini açıklayan değişkenleri) listesine değişkenler eklenir (ör. `inflation`, `interest`, `exchange`). Bu sıra, Cholesky şoklarına dayalı IRF ve FEVD'de kullanılan sıradır (Bölüm 10.7).
    *   **Lag order** kutusuna seçilen gecikme (ör. 2) yazılır.
    *   Deterministik terimler (sabit, trend, mevsimsel kuklalar) işaretlenir.
    *   **OK** dendiğinde her denklem için katsayılar ve **F-tests of zero restrictions** tabloları gösterilir.
3.  **Sonuç penceresinden:**
    *   **Graphs → VAR inverse roots** ile ters köklerin (yani öz değerlerin) birim çember içindeki konumu (stabilite, Bölüm 10.5 ve Şekil 10.2),
    *   **Graphs → Impulse responses (combined)** ya da **Analysis → Impulse responses** ile etki-tepki fonksiyonları,
    *   **Analysis → Forecast variance decomposition** ile FEVD tabloları,
    *   **Tests → Autocorrelation** ile artıklarda otokorelasyon testi,
    *   **Analysis → Forecasts…** ile tahminler elde edilir.

**Granger nedenselliği nerede?** Gretl VAR çıktısında her denklemin altında yer alan **"F-tests of zero restrictions"** bölümündeki "All lags of interest" satırı, enflasyon denkleminde faizin tüm gecikme katsayılarının sıfır olduğu hipotezini sınar. Bu, Bölüm 10.6'daki Granger nedensellik testinin ta kendisidir: p-değeri 0.05'ten küçükse faiz, enflasyonun Granger nedenidir. Bu veride satır `F(2, 111) = 0.17759 [0.8375]` şeklindedir: köşeli parantezdeki 0.8375 p-değeridir ve faiz enflasyonun Granger nedeni değildir. Aynı denklemdeki "All lags of exchange" satırı ise `F(2, 111) = 33.412 [0.0000]` verir.

> **Hemen deneyin 11.6.1a · Gretl'de hazır bir veriyle VAR ve ters kökler** (Gretl menüleri, ~5 dk)
>
> Yukarıdaki menü adımlarını önce Gretl'in hazır örnek verisiyle kısa bir denemede uygulayalım.
>
> 1. **File → Open data → Sample file…** seçin; açılan pencerenin **Gretl** sekmesinde `denmark` dosyasına çift tıklayın (Danimarka, 1974–1987, üç aylık: `LRM` log reel para, `LRY` log reel gelir, `IBO` tahvil faizi, `IDE` mevduat faizi).
> 2. **Model → Multivariate time series → VAR lag selection…** seçin; dört değişkeni de listeye ekleyin, en büyük gecikmeyi **4** yapıp **OK**'a basın.
> 3. **Model → Multivariate time series → Vector Autoregression…** ile aynı dört değişkenle **Lag order** = 2 olan modeli kurun. Sonuç penceresinde **Graphs → VAR inverse roots** seçin.
>
> Aynı işin betik karşılığı (**File → Script files → New script** ile açılan pencereye yazıp çalıştırabilirsiniz):
>
> ```gretl
> open denmark
> var 4 LRM LRY IBO IDE --lagselect
> var 2 LRM LRY IBO IDE --quiet
> eval abs(eigen($compan))     # eşlik matrisinin öz değerlerinin modülleri
> ```
>
> `$compan` son kurulan VAR'ın eşlik matrisidir (10.5), `eigen()` öz değerlerini, `abs()` modüllerini verir.
>
> **Ne göreceksiniz:** Gecikme tablosunda yıldızlar AIC ve HQC için 2. satırda, BIC için 1. satırdadır:
>
> ```text
> lags        loglik    p(LR)       AIC          BIC          HQC
>
>    1     609.15315           -23.104045   -22.346466*  -22.814552 
>    2     631.70153  0.00013  -23.360844*  -21.997203   -22.839757*
>    3     642.38574  0.16478  -23.152382   -21.182677   -22.399699 
>    4     653.22564  0.15383  -22.950025   -20.374257   -21.965748 
> ```
>
> Ters kök grafiğinde 8 nokta birim çemberin içindedir; betiğin son satırı modülleri 0.398, 0.175, 0.602, 0.602, 0.509, 0.810, 0.810, 0.966 olarak yazar. En büyüğü 0.966'dır: Model stabildir ama `macro.csv` (Bölüm 10.9) ve `Canada` (Hemen deneyin 10.9b) modellerindeki gibi sınıra yakındır.
>
> **Şimdi siz:** VAR(2) çıktısında `IBO` (tahvil faizi) denkleminin altındaki **F-tests of zero restrictions** bölümüne bakın. Hangi değişkenlerin gecikmeleri tahvil faizini öngörmeye yardım ediyor (p < 0.05)?
>
> <details><summary>Cevap</summary>
>
> Yalnızca kendi gecikmeleri: `All lags of IBO  F(2, 44) = 55.884 [0.0000]`. `LRM` (p = 0.9978), `LRY` (p = 0.0653) ve `IDE` (p = 0.7731) satırları 0.05'in üstündedir; bu değişkenler tahvil faizinin Granger nedeni değildir (`LRY` sınıra yakındır). Ters yönde ise `LRM` denkleminde `All lags of IBO` satırı p = 0.0006 verir: Tahvil faizi, para talebini öngörmeye yardım eder.
>
> </details>

> **Hemen deneyin 11.6.1b · Değişken sırası değişince IRF ve FEVD değişir mi?** (Gretl menüleri ya da konsol, ~5 dk)
>
> Cholesky sıralamasının etkisini (Bölüm 10.7) görelim. `data/macro.csv` dosyasını **File → Open data → User file…** ile açın. **Model → Multivariate time series → Vector Autoregression…** seçin; **Endogenous variables** listesine değişkenleri bu kez `exchange`, `interest`, `inflation` sırasıyla ekleyin, **Lag order** kutusuna 2 yazıp **OK** deyin. Sonuç penceresinde **Analysis → Impulse responses** ve **Analysis → Forecast variance decomposition** tablolarını açın. Konsolda aynı iş:
>
> ```gretl
> set horizon 12
> var 2 exchange interest inflation --impulse-responses --variance-decomp
> ```
>
> **Ne göreceksiniz:** Katsayılar, Granger testleri ve FEVD tablosundaki `std. error` sütunu değişmez; değişen yalnızca aynı aydaki şokların değişkenlere nasıl paylaştırıldığıdır. Kur şokuna enflasyonun 1. dönemdeki tepkisi artık 0 değil 0.127'dir; faiz şokuna tepkisi de −0.109'dur. Kur şokuna enflasyonun en yüksek tepkisi yine 5. dönemdedir (4.42 yerine 4.55). Enflasyonun 12. dönem FEVD'si %44.0 kendi, %0.8 faiz, %55.2 kur yerine %38.5 kendi, %4.4 faiz, %57.1 kur olur.
>
> **Şimdi siz:** Faizin 12. dönem FEVD'sinde kur şokunun payı iki sıralamada kaçtır? Neden değişti?
>
> <details><summary>Cevap</summary>
>
> `inflation interest exchange` sırasında %10.5, `exchange interest inflation` sırasında %19.7. Kur listenin başına alınınca, faiz ve kur şoklarının aynı aydaki ortak kısmı (artık korelasyonu 0.18, Bölüm 10.9.1) tümüyle kura yazılır. Veri hangi şokun önce geldiğini söyleyemez; sıralama bir varsayımdır ve raporlanmalıdır.
>
> </details>

#### 11.6.2. Betik ile VAR

```gretl
# Gretl ile VAR örneği (tam betik ve ayrıntılı yorumlar: Codes/gretl/ch11_var.inp)

# 1) Veri (depo kök dizininden) ve zaman serisi yapısı
open "data/macro.csv"
setobs 12 2015:01 --time-series

# 2) Gecikme seçimi
var 8 inflation interest exchange --lagselect

# 3) VAR(2), IRF ve FEVD (değişken sırası = Cholesky sırası)
set horizon 12
var 2 inflation interest exchange --impulse-responses --variance-decomp

# 4) Artık tanı testleri
modtest --autocorr
modtest --normality

# 5) 12 aylık örnek dışı tahmin
dataset addobs 12
fcast --out-of-sample
```

Dosya: [`Codes/gretl/ch11_var.inp`](../Codes/gretl/ch11_var.inp) (Gretl'de **File → Script files → Open user file…** ile açıp çalıştırabilirsiniz).

Betikteki komutlar şöyle çalışır: `var 8 inflation interest exchange --lagselect` 1'den 8'e kadar her gecikme için bir VAR kurar ama sonuçlarını göstermez; yalnızca log-olabilirlik, olabilirlik oranı testi p-değeri (`p(LR)`) ve AIC/BIC/HQC tablosunu yazar. `set horizon 12` IRF ve FEVD tablolarının kaç dönem ileriye kadar hesaplanacağını ayarlar. `var 2 ...` asıl modeli kurar; ilk sayı gecikme sayısı, ardından gelen liste içsel değişkenlerdir ve listedeki sıra Cholesky sırasıdır. `--impulse-responses` ve `--variance-decomp` IRF ve FEVD tablolarını model çıktısının altına ekler. `modtest` son kurulan modele uygulanır: VAR'da `--autocorr` tek denklemlik Ljung-Box yerine tüm denklemleri birlikte sınayan Rao F testini, `--normality` ise Doornik-Hansen testini yapar. `fcast --out-of-sample` yalnızca örnek dışı (2025) dönemi için tahmin üretir.

Bu betik çalıştırıldığında Gretl önce gecikme seçim tablosunu, ardından VAR sonuç tablosunu (Granger testleri dahil), IRF ve FEVD tablolarını, artık testlerini ve tahminleri sırasıyla yazdırır. Sonuçlar Bölüm 10.9'daki Python sonuçlarıyla karşılaştırılabilir:

*   **Gecikme seçimi:** AIC 7, BIC 2, HQC 3 (aynı). Gretl'in kriter değerleri statsmodels'tan farklı bir sabit içerdiği için sayılar farklıdır (ör. BIC 10.82'ye karşı 2.30), ama seçilen gecikmeler aynıdır.
*   **Katsayılar:** Birebir aynıdır (ör. enflasyon denkleminde `exchange_1` = 3.667, `exchange_2` = −3.718). Gretl gecikmeleri `inflation_1`, `inflation_2` biçiminde, statsmodels `L1.inflation`, `L2.inflation` biçiminde adlandırır. Her denklemin altındaki Durbin-Watson değerleri de aynıdır (2.04, 2.23, 1.83).
*   **Granger (F-tests of zero restrictions):** faiz → enflasyon p = 0.84, döviz kuru → enflasyon p < 0.001, enflasyon → faiz p = 0.022, döviz kuru → faiz p = 0.0002.
*   **IRF:** Gretl'in tabloları "one-standard error shock", yani Cholesky şoklarına göredir ve 1. dönem şokun geldiği andır (Python'da $h = 0$). Enflasyon listede ilk sırada olduğu için faiz ve kur şoklarına enflasyonun 1. dönemdeki tepkisi 0'dır (Bölüm 10.7). Kur şokuna enflasyonun tepkisi 5. dönemde en yüksek değerine (4.42) ulaşır. Python'un `orth=True` tepkileri yaklaşık %3 büyük çıkar, çünkü statsmodels şok büyüklüğünü serbestlik derecesine göre düzeltilmiş varyansla, Gretl ise düzeltmesiz varyansla hesaplar.
*   **FEVD (12. dönem, enflasyon):** %44.0 kendi, %0.8 faiz, %55.2 döviz kuru şoku (aynı).
*   **Artık testleri:** Rao F otokorelasyon testi tüm gecikmelerde p < 0.01 verir; Doornik-Hansen normallik testi de reddeder. Python'daki portmanteau testiyle aynı sonuç: düzey serilerle kurulan model artıklarda yapı bırakmaktadır.
*   **Tahmin:** 2025 Ocak için enflasyon 42.23, faiz 48.87, kur 35.47 (Python'la aynı). Tablodaki %95 aralıklar ufuk uzadıkça hızla genişler; Aralık 2025 enflasyon tahmini için aralık yaklaşık 13 ile 77 arasındadır.

IRF grafiklerini ve ters kök grafiğini görmek için betik çalıştıktan sonra model penceresindeki **Graphs** menüsü kullanılabilir.

**Not —** Bölüm 10'daki Python uygulamasında olduğu gibi, VAR'a girmeden önce serilerin durağanlığı ADF/KPSS ile kontrol edilmeli (Bölüm 11.4), gerekirse **Add → First differences of selected variables** ile farkları alınmalıdır. Betikte bunu denemek için `var` satırlarından önce `diff inflation interest exchange` yazıp modeli `d_inflation d_interest d_exchange` değişkenleriyle kurabilirsiniz.

Durağanlık kontrolünün ve stabilitenin Gretl'deki karşılığı şöyle yazılabilir:

```gretl
diff inflation interest exchange
adf 12 inflation --c --test-down=AIC      # en çok 12 gecikme, sayıyı AIC seçer
adf 12 d_inflation --c --test-down=AIC
var 2 inflation interest exchange --quiet
eval maxc(abs(eigen($compan)))            # en büyük öz değer modülü
```

`--test-down=AIC` seçeneği verildiğinde `12` sabit gecikme değil üst sınırdır: Gretl 12 gecikmeden başlayıp gecikme sayısını AIC ile geriye doğru seçer. Bu veride düzey serilerin p-değerleri 0.9158 (enflasyon), 0.1278 (faiz) ve 1 (kur) çıkar; farkı alınmış serilerde 0.0153, 0.0142 ve 1.2e-08'e iner. Gretl gecikmeyi statsmodels'tan farklı seçtiği için düzey enflasyonun p-değeri Bölüm 10.9'daki 0.806'dan farklıdır, ama karar aynıdır: düzey seriler durağan değil, farklar durağan. Son satır VAR(2) için 0.99737 yazar (Şekil 10.2b'deki 0.997); aynı satırı `var 7 ...` sonrasında çalıştırırsanız 1.0238 çıkar ve AIC'nin önerdiği VAR(7)'nin stabil olmadığını görürsünüz (Bölüm 10.9).

> **Hemen deneyin 11.6.2 · Farkı alınmış serilerle VAR: Granger sonuçları değişir mi?** (Gretl betiği, ~5 dk)
>
> Yukarıdaki notu deneyelim. **File → Script files → New script** ile yeni bir betik açın, aşağıdakileri yapıştırıp **Run** deyin (Gretl dosyayı bulamazsa `open` satırına `macro.csv`'nin tam yolunu yazın).
>
> ```gretl
> open "data/macro.csv"
> setobs 12 2015:01 --time-series
> diff inflation interest exchange     # d_inflation, d_interest, d_exchange
> var 8 d_inflation d_interest d_exchange --lagselect
> var 2 d_inflation d_interest d_exchange
> modtest --autocorr                   # Rao F otokorelasyon testi
> ```
>
> **Ne göreceksiniz:** Gecikme tablosunda AIC'nin yıldızı **8**'de, BIC ve HQC'nin yıldızı **1**'dedir. VAR(2) çıktısında `d_inflation` denkleminin altında `All lags of d_interest F(2, 110) = 0.22847 [0.7961]` (faiz → enflasyon yine Granger nedeni değil; düzeyde 0.8375'ti) ve `All lags of d_exchange F(2, 110) = 20.934 [0.0000]` (kur → enflasyon yine çok anlamlı) yazar. Otokorelasyon testinde 1–4. gecikmelerin p-değerleri 0.2257, 0.3827, 0.3187, 0.2416'dır (hepsi > 0.05); 5. gecikmeden itibaren p < 0.05 olur. Düzey serilerle kurulan VAR(2)'de bütün gecikmelerde p < 0.01 idi.
>
> **Şimdi siz:** Aynı çıktıda `d_interest` denkleminde `All lags of d_inflation` satırının p-değeri kaçtır? Düzey modelde enflasyon → faiz p-değeri 0.022'ydi (anlamlı).
>
> <details><summary>Cevap</summary>
>
> `All lags of d_inflation F(2, 110) = 0.46367 [0.6302]`: farkı alınmış modelde enflasyon artık faizin Granger nedeni görünmez. Python'da (Hemen deneyin 10.6) gördüğümüz sonucun aynısıdır. Kur → faiz ilişkisi ise farklarla da anlamlı kalır (`All lags of d_exchange F(2, 110) = 4.2257 [0.0171]`).
>
> </details>

---

### 11.7. Gretl'in Ekosistemdeki Yeri

Toparlamak için ders boyunca kullanılan araçları şöyle karşılaştırabiliriz:

| Araç | Güçlü yönleri |
| --- | --- |
| **Gretl** | OLS, ARIMA, VAR gibi klasik ekonometrik modellerin hızlıca denenmesi; grafik arayüz ile komut dilinin bir arada olması; hazır tanı testleri. |
| **R / Python** | Esnek veri işleme ve otomasyon; Prophet (Bölüm 9), XGBoost (Bölüm 13), LSTM/GRU/1D-CNN (Bölüm 15) gibi gelişmiş modeller. |
| **Weka** | Kod yazmadan makine öğrenmesi algoritmalarını denemek; zaman serilerini gecikmeli değişkenlerle tabloya dönüştürüp regresyon uygulamak (Bölüm 13 ve 14). |

Gretl bu resmin içinde, özellikle zaman serisi ve ekonometrik modellerin temel mantığını görmek için oldukça işlevli bir araçtır. Aynı veriyi Gretl, Python ve Weka'da çalıştırmak, hem yöntemleri hem de ortamların farklarını karşılaştırmak için iyi bir alıştırmadır.

> **Hemen deneyin 11.7 · Aynı model iki programda: Gretl ve R** (R ve Gretl, ~4 dk)
>
> 11.3.3'te Gretl'de kurduğunuz sabitsiz airline modelin sonuçlarını hatırlayın: `theta_1 = -0.401823`, `Theta_1 = -0.556937`, `Log-likelihood 244.6965`, `Akaike criterion -483.3930`. Şimdi RStudio konsolunda aynı modeli kurun:
>
> ```r
> library(forecast)
> m <- Arima(log(AirPassengers), order = c(0, 1, 1), seasonal = c(0, 1, 1))
> m      # katsayılar, log-olabilirlik ve AIC
> ```
>
> **Ne göreceksiniz:** `ma1 = -0.4018`, `sma1 = -0.5569`, `log likelihood = 244.7`, `AIC=-483.4`. Sonuçlar Gretl'inkilerle aynıdır; yalnızca adlar farklıdır (Gretl'in `theta_1` / `Theta_1`'i, R'da `ma1` / `sma1`). Dördüncü basamaktaki küçük farklar iki programın sayısal optimizasyonunun farklı noktalarda durmasından kaynaklanır.
>
> **Şimdi siz:** R'da `exp(forecast(m, h = 12)$mean)` yazarak 1961 tahminlerini yolcu sayısına çevirin. Ocak ve Temmuz 1961 değerlerini 11.5'teki Gretl sonuçlarıyla (yaklaşık 450 ve 670) karşılaştırın.
>
> <details><summary>Cevap</summary>
>
> R'da Ocak 1961 için **450.4**, Temmuz 1961 için **670.0** çıkar; Gretl betiğinin `pass_f` değerleriyle aynıdır. Aynı modeli iki programda çalıştırıp sonuçların tutmasını görmek, hem kendi kodunuzu hem de programların ayarlarını (ör. Gretl'de `--nc`) kontrol etmenin iyi bir yoludur.
>
> </details>

---

[İçindekiler](../Course_notes.md) · [← 10. VAR: Çok Değişkenli Zaman Serisi Modeli](10_var_cok_degiskenli_zaman_serisi_modeli.md) · [12. Yapay Zeka ile Zaman Serisi Analizine Giriş →](12_yapay_zeka_ile_zaman_serisi_analizine_giris.md)
