# **Gretl ile Uygulamalı Ekonometri: Zaman Serisi Tahmini ve Anomali Tespiti Kılavuzu**

## **Giriş: Veri Analitiğine Yolculuk ve Gretl Felsefesi**

Hedefimiz, açık kaynak kodlu ve kullanıcı dostu bir araç olan Gretl (Gnu Regression, Econometrics and Time-series Library \- Gnu Regresyon, Ekonometri ve Zaman Serisi Kütüphanesi) yazılımını kullanarak, zaman serisi analizinin derinliklerine inmektir.1

İlk olarak, ekonometri literatürünün klasiklerinden biri olan AirPassengers veri setini ele alarak, geçmiş yolcu verilerinden geleceğe dair tutarlı tahminler (Forecasts) üretmeyi öğreneceğiz. İkinci bölümde ise, finansal verilerden teknolojik altyapı verilerine geçiş yaparak, bir sunucunun CPU (Central Processing Unit \- Merkezi İşlem Birimi) kullanım verilerindeki anormallikleri (Anomaly Detection) tespit edeceğiz. 

**Neden Gretl? Açık Kaynak Felsefesi ve Kullanıcı Dostu Arayüz**

Veri biliminde R (Programming Language R \- R Programlama Dili) ve Python gibi araçlar hakimiyetini korusa da, özellikle zaman serisi analizi ve ekonometrik modelleme söz konusu olduğunda Gretl, benzersiz bir konuma sahiptir. Gretl, R'ın istatistiksel gücünü, ticari yazılımların sunduğu kullanıcı dostu GUI (Graphical User Interface \- Grafiksel Kullanıcı Arayüzü) ile birleştirir.1 Yeni başlayanlar için komut satırı ekranında (Command Line Interface) kaybolmadan, menüler aracılığıyla karmaşık modelleri kurmak büyük bir avantajdır. Ancak Gretl sadece bir "tıkla-ve-gör" yazılımı değildir; arka planda çalışan güçlü algoritmaları ve hansl (Hancock's Script Language) adı verilen senaryo dili ile ileri düzey otomasyona da imkan tanır.2

Gretl'in en önemli özelliklerinden biri, şeffaflığıdır. Ticari yazılımların aksine, bir tahminin nasıl üretildiği veya bir test istatistiğinin nasıl hesaplandığı tamamen açıktır. Bu ders notlarında, Gretl'in bu şeffaf yapısından faydalanarak, analizlerimizin her adımını doğrulayacağız.

## ---

**Bölüm 1: Zaman Serisi Analizinin Temelleri**

Uygulamaya geçmeden önce, üzerinde çalışacağımız "zaman serisi" kavramını ve analizimizin teorik iskeletini anlamak zorundayız. Bir zaman serisi, kronolojik olarak sıralanmış veri noktaları dizisidir. Bu, bir hisse senedinin günlük kapanış fiyatı, bir şehrin yıllık yağış miktarı veya bizim örneğimizdeki gibi aylık havayolu yolcu sayısı olabilir. Zaman serisi analizinin temel amacı, verinin geçmişteki davranış kalıplarını (Pattern) modelleyerek, bu kalıpların gelecekte de devam edeceği varsayımıyla öngörülerde bulunmaktır.

### **1.1. Zaman Serisinin Bileşenleri**

Bir zaman serisi grafiğine baktığımızda gördüğümüz dalgalanmalar rastgele değildir; genellikle dört ana bileşenin birleşimidir 3:

1. **Trend (Eğilim):** Verinin uzun vadede yukarı veya aşağı yönlü hareketidir. Örneğin, havacılık sektörünün gelişmesiyle birlikte yolcu sayılarının yıllar içinde sürekli artması bir trenddir.  
2. **Seasonality (Mevsimsellik):** Belirli bir zaman dilimi içinde (bir yıl, bir hafta, bir gün) düzenli olarak tekrarlanan hareketlerdir. Yaz aylarında tatil nedeniyle uçuşların artması, kışın azalması tipik bir mevsimsellik örneğidir.  
3. **Cyclicality (Döngüsellik):** Mevsimsel olmayan ancak ekonomik konjonktüre (örneğin krizler veya büyüme dönemleri) bağlı olarak ortaya çıkan, periyodu sabit olmayan dalgalanmalardır.  
4. **Irregular/Random Component (Düzensiz/Rastgele Bileşen):** Hata terimi (Error Term) veya artık (Residual) olarak da adlandırılır. Modelin açıklayamadığı, öngörülemeyen şokları temsil eder.

Analizimizin başarısı, bu bileşenleri ne kadar iyi ayrıştırabildiğimize (Decomposition) bağlıdır. Gretl, bu ayrıştırma işlemini hem görsel hem de istatistiksel olarak yapmamıza olanak tanıyan araçlarla doludur.

### **1.2. Durağanlık (Stationarity) Kavramı**

Zaman serisi analizinin "oturduğu temel" durağanlıktır. Bir serinin istatistiksel özelliklerinin (ortalama, varyans, otokorelasyon) zamanla değişmemesi durumuna durağanlık denir. Neden bu kadar önemlidir? Çünkü biz, geçmişteki ilişkilerin gelecekte de geçerli olacağını varsayarak model kurarız. Eğer serinin ortalaması sürekli değişiyorsa (yani trend varsa) veya varyansı zamanla artıyorsa, kurduğumuz model gelecekte geçerliliğini yitirecektir.

AirPassengers verisinde göreceğimiz gibi, çoğu gerçek hayat verisi durağan değildir. Bu nedenle analiz sürecimizin büyük bir kısmı, veriyi durağan hale getirmek (Transformation) üzerine kurgulanacaktır. Bu dönüşümler genellikle logaritma alma (varyansı sabitlemek için) ve fark alma (Differencing \- ortalamayı sabitlemek için) işlemlerini içerir.

## ---

**Bölüm 2: Modül 1 \- AirPassengers Verisi ile Gelecek Tahmini**

Bu modülde, Box ve Jenkins'in meşhur "Airline Data" (Havayolu Verisi) setini kullanarak adım adım bir tahmin modeli inşa edeceğiz. Hedefimiz, 1960 yılı sonunda biten veri setini kullanarak, 1961 yılının ilk 10 ayı için yolcu sayılarını tahmin etmektir.

### **2.1. Adım 1: Veri Setine Erişim ve İçe Aktarma**

Gretl, eğitim amaçlı kullanım için geniş bir örnek veri kütüphanesi sunar. AirPassengers verisi bu kütüphanenin en popüler parçalarından biridir. Veriyi dışarıdan indirmekle uğraşmak yerine, doğrudan Gretl sunucularından çekeceğiz.

**Uygulama Adımları:**

1. Bilgisayarınızda Gretl programını başlatın.  
2. Ana menü çubuğundan **File (Dosya)** \> **Open data (Veri aç)** \> **Sample file (Örnek dosya)** yolunu izleyin.5  
3. Karşınıza çıkan pencerede sekmeler göreceksiniz. **Gretl** sekmesine tıklayın.  
4. Bu pencere, Gretl ile birlikte gelen yerel dosyaları gösterir. Ancak AirPassengers bazen sunucuda tutulur. Pencerenin sol üst veya alt kısmında bulunan **Look on server (Sunucuda ara)** butonuna tıklayın.5 Bu işlem, internet bağlantınızı kullanarak Gretl'in global veri havuzuna bağlanır.  
5. Açılan listede dosya kodlarına veya açıklamalarına göz atın. Aradığımız dosyanın kodu genellikle **bjg** (Box-Jenkins Gas/Airline) veya doğrudan **AirPassengers** olarak listelenir. Açıklama kısmında "Monthly airline passenger totals 1949-1960" ibaresini görmelisiniz.  
6. Dosyayı seçin ve **Install (Yükle)** butonuna tıklayın. Veri bilgisayarınıza indikten sonra, açmak için üzerine çift tıklayın veya **Open (Aç)** deyin.

Veri yüklendiğinde, Gretl'in ana penceresinde g (veya AirPassengers) isimli bir değişken belirecektir. Ayrıca pencerenin alt kısmında verinin frekansının "Monthly" (Aylık) olduğunu ve 1949:01 ile 1960:12 arasında 144 gözlem içerdiğini teyit edebilirsiniz.

### **2.2. Adım 2: Keşifsel Veri Analizi (EDA) ve Görselleştirme**

Veriyi modele sokmadan önce "görmek" hayati önem taşır. İnsan beyni, grafiklerdeki örüntüleri (pattern) algılamada matematiksel testlerden daha hızlı olabilir.

**Görselleştirme Uygulaması:**

1. Ana penceredeki g değişkeninin üzerine fare ile bir kez tıklayarak seçili hale getirin.  
2. Sağ tıklayın ve açılan menüden **Time series plot (Zaman serisi grafiği)** seçeneğini işaretleyin.3 Alternatif olarak menüden **View (Görünüm)** \> **Graph specified vars (Belirtilen değişkenlerin grafiği)** \> **Time series plot** yolunu da kullanabilirsiniz.

Grafiğin Yorumlanması (İçgörü):  
Ekrana gelen grafiği dikkatlice incelediğimizde üç temel olgu göze çarpar:

* **Belirgin Pozitif Trend:** Çizgi sol alttan sağ üste doğru sürekli bir tırmanış halindedir. Bu, verinin ortalamasının sabit olmadığını (durağan olmadığını) ve "birinci momentte" (first moment) bir hareket olduğunu gösterir.  
* **Güçlü Mevsimsellik:** Her yılın belli dönemlerinde zirve yapan (muhtemelen yaz ayları), belli dönemlerinde düşen düzenli dalgalanmalar vardır.  
* **Artan Varyans (Heteroskedasticity \- Değişen Varyans):** Bu nokta çok kritiktir. Grafiğin sol tarafındaki (1950'ler başı) dalgaların boyu ile sağ tarafındaki (1960'lar) dalgaların boyu aynı değildir. Yolcu sayısı arttıkça, mevsimsel dalgalanmaların şiddeti de artmaktadır. Bu durum, verinin **Toplamsal (Additive)** değil, **Çarpımsal (Multiplicative)** bir yapıya sahip olduğunu gösterir. Yani mevsimsellik, trendden bağımsız sabit bir değer değil, trendin bir yüzdesi (katsayısı) gibi davranmaktadır.

### **2.3. Adım 3: Veri Dönüşümleri (Transformations)**

Model kurarken "Çarpımsal" yapıyı "Toplamsal" yapıya çevirmek ve varyansı sabitlemek için standart prosedür, verinin doğal logaritmasını almaktır. Logaritma dönüşümü, üstel büyümeyi lineer hale getirir ve artan dalgalanmaları bastırır.

**Logaritma Alma İşlemi:**

1. g değişkeni seçiliyken, menüden **Add (Ekle)** \> **Logs of selected variables (Seçili değişkenlerin logaritması)** seçeneğine tıklayın.  
2. Kısa yol olarak, değişkene sağ tıklayıp **Add log (Logaritma ekle)** diyebilirsiniz.  
3. Ana pencereye l\_g (log of g) adında yeni bir değişken eklenecektir.

Şimdi l\_g değişkeninin grafiğini çizerseniz, dalgalanmaların genliğinin (amplitude) zaman içinde daha stabil hale geldiğini göreceksiniz. Artık varyans durağanlığına yaklaştık, ancak hala bir trendimiz (ortalama durağan değil) ve mevsimselliğimiz var. Bunları model aşamasında "fark alma" (Differencing) yöntemiyle çözeceğiz.

### **2.4. Adım 4: ARIMA Modelinin Belirlenmesi**

Zaman serisi öngörülerinde endüstri standardı olan **ARIMA (AutoRegressive Integrated Moving Average \- Otoregresif Bütünleşik Hareketli Ortalama)** modellerini kullanacağız. Verimiz mevsimsel olduğu için bu, **SARIMA (Seasonal ARIMA)** formuna dönüşecektir.

Bir SARIMA modeli $(p, d, q) \\times (P, D, Q)\_s$ parametreleri ile tanımlanır:

* **Trend Bileşenleri:**  
  * $p$ (AR \- AutoRegressive): Geçmiş değerlerin bugüne etkisi.  
  * $d$ (I \- Integrated): Trendi yok etmek için alınan fark sayısı.  
  * $q$ (MA \- Moving Average): Geçmiş hata terimlerinin bugüne etkisi.  
* **Mevsimsel Bileşenler:**  
  * $P$ (Seasonal AR): Mevsimsel geçmiş değerlerin etkisi (örneğin geçen senenin aynı ayı).  
  * $D$ (Seasonal I): Mevsimsel fark sayısı.  
  * $Q$ (Seasonal MA): Mevsimsel hata terimlerinin etkisi.  
  * $s$: Mevsimsellik periyodu (Aylık veri için 12).

Model Seçimi (Identification):  
Hangi parametrelerin seçileceği normalde ACF (Autocorrelation Function \- Otokorelasyon Fonksiyonu) ve PACF (Partial Autocorrelation Function \- Kısmi Otokorelasyon Fonksiyonu) grafikleri (korelogramlar) incelenerek belirlenir. Ancak AirPassengers verisi için literatürde kabul görmüş, Box ve Jenkins tarafından önerilen "Airline Model" yapısı şöyledir: ARIMA(0,1,1)(0,1,1)12.  
Bu yapının mantığı şudur:

* **Logaritmik Veri:** Varyansı sabitlemek için.  
* **$d=1$:** Trendi yok etmek için serinin birinci farkı alınır ($Y\_t \- Y\_{t-1}$).  
* **$D=1$:** Mevsimselliği yok etmek için serinin 12\. dereceden (mevsimsel) farkı alınır ($Y\_t \- Y\_{t-12}$).  
* **$q=1$ ve $Q=1$:** Hem kısa vadeli hem de mevsimsel düzeltmeler için birer adet Hareketli Ortalama (MA) terimi eklenir.

Gretl'de bu modeli nasıl kuracağımızı görelim.

### **2.5. Adım 5: Modelin Tahmini (Estimation)**

Gretl'in arayüzü, ARIMA modellerini tanımlamayı son derece basit hale getirir.

**Uygulama:**

1. Menüden **Model** \> **Time series (Zaman serisi)** \> **ARIMA...** yolunu izleyin.6  
2. Açılan "ARIMA specifications" penceresinde ayarları şu şekilde yapın:  
   * **Dependent variable (Bağımlı değişken):** l\_g değişkenini seçin (DİKKAT: Orijinal g verisini değil, logaritması alınmış veriyi seçmelisiniz).  
   * **Non-seasonal (Mevsimsel olmayan) parameters:**  
     * AR: 0  
     * Integrated (Fark): 1  
     * MA: 1  
   * **Seasonal (Mevsimsel) parameters:**  
     * AR: 0  
     * Integrated (Fark): 1  
     * MA: 1  
   * **Seasonality (Mevsimsellik periyodu):** 12 (Gretl genellikle verinin frekansına bakarak bunu otomatik önerir).  
   * **Include constant (Sabit terim ekle):** Genellikle çift fark alma işleminde ($d=1, D=1$) sabit terim gereksiz hale gelir, ancak başlangıçta seçili bırakıp istatistiksel anlamlılığını kontrol etmek iyi bir pratiktir.  
3. **OK** butonuna tıklayarak modeli çalıştırın.

Sonuçların Yorumlanması:  
Karşınıza "Model 1: ARIMA, using observations 1950:02-1960:12" başlıklı bir pencere gelecektir. Burada odaklanmanız gereken noktalar:

* **Katsayılar (Coefficients):** theta\_1 (MA1) ve Theta\_1 (Seasonal MA1) satırlarına bakın. "p-value" (olasılık değeri) sütununda değerler 0.05'ten küçükse (genellikle \*\*\* veya \*\* ile işaretlenir), bu değişkenler model için anlamlıdır. Airline modelinde her iki MA teriminin de yüksek derecede anlamlı olması beklenir.  
* **Bilgi Kriterleri (AIC/BIC):** Pencerenin alt kısmında **AIC (Akaike Information Criterion)** ve **BIC (Bayesian Information Criterion)** değerlerini göreceksiniz.7 Bu değerler "ceza puanı" gibidir; ne kadar düşükse model o kadar iyidir. Farklı modeller denerseniz (örneğin AR(1) eklemek gibi), bu değerleri karşılaştırarak en iyi modeli seçebilirsiniz.

### **2.6. Adım 6: Gelecek Tahmini (Forecasting) ve Hazırlık**

Gretl'de geleceğe yönelik tahmin yapmanın püf noktası, veri setinin zaman eksenini "genişletmektir". Gretl, sadece veri setinde tanımlı olan satırlar için işlem yapar. Eğer veri setiniz 1960 Aralık'ta bitiyorsa ve siz 1961'i tahmin etmek istiyorsanız, öncelikle Gretl'e "Bana 1961 için boş yer aç" demelisiniz.

**Gözlem Ekleme (Adding Observations):**

1. Ana pencereden **Data (Veri)** \> **Add observations... (Gözlem ekle...)** menüsünü seçin.1  
2. Açılan kutucuğa, kaç dönem ileriye gitmek istediğinizi yazın. Hedefimiz 10 aylık tahmin olduğu için **10** yazıyoruz.  
3. **OK** dediğinizde, veri setinizin sonuna 10 adet yeni satır eklendiğini göreceksiniz. Gretl, zaman serisi yapısını tanıdığı için bu yeni satırları otomatik olarak **1961:01**'den **1961:10**'a kadar tarihlendirir. Ancak bu satırlardaki g ve l\_g değerleri şimdilik boştur (Missing Value).

**Tahmin İşlemi:**

1. ARIMA sonuç penceresine geri dönün (Eğer kapattıysanız, ana pencerenin altındaki "Session icons" kısmından dişli çark ikonuna tıklayarak veya Model 1 tablosuna çift tıklayarak açabilirsiniz).  
2. Bu pencerede **Analysis (Analiz)** \> **Forecasts... (Öngörüler...)** seçeneğine tıklayın.8  
3. Karşınıza "Forecast" ayar penceresi gelecektir:  
   * **Start of forecast range:** Otomatik olarak örneklem dışı ilk tarih (1961:01) seçili gelir.  
   * **Length of forecast:** Eklediğiniz gözlem sayısı (10) kadar ayarlanır.  
   * **Pre-forecast observations:** Grafikte geçmiş verinin ne kadarının gösterileceğini belirler. "24" seçerseniz, tahminin arkasına son 2 yılın gerçek verisini ekleyerek görsel bütünlük sağlar.  
   * **Create new variable:** Tahmin sonuçlarını kaydetmek için bir isim verin (örneğin forecast\_yolcu).  
   * **Standard error plots:** Bu kutucuğu işaretlemek çok önemlidir. Bu, tahminin güven aralıklarını (Confidence Intervals) gösteren **"Fan Chart" (Yelpaze Grafiği)** oluşturulmasını sağlar.10  
4. **OK** butonuna basın.

### **2.7. Adım 7: Sonuçların ve Fan Chart'ın Analizi**

Ekrana gelen grafik, yaptığımız çalışmanın meyvesidir.

* **Ortadaki Çizgi (Point Forecast):** Modelin, 1961 yılının ilk 10 ayı için en olası gördüğü yolcu sayılarıdır. Model mevsimselliği başarıyla yakaladığı için, tahmin çizgisinin de tıpkı geçmiş yıllardaki gibi dalgalandığını (yazın artıp sonbaharda düştüğünü) göreceksiniz.  
* **Gölgeli Alanlar (Güven Aralığı):** Genellikle %95 güven aralığını temsil eder. Gelecek, tanımı gereği belirsizdir. Zamanda ileri gittikçe (Ocak'tan Ekim'e doğru), bu belirsizlik artar. Bu nedenle güven aralığı, bir yelpaze gibi genişler. İşte buna literatürde **Fan Chart** denir.11 Grafiğin anlamı şudur: "Ekim 1961 için tam rakamı bilemeyiz, ancak %95 ihtimalle yolcu sayısı şu alt sınır ile şu üst sınır arasında olacaktır."

Önemli Bir Detay: Geri Dönüşüm (Back-Transformation)  
Modeli l\_g (logaritmik veri) üzerine kurmuştuk. Dolayısıyla üretilen ham tahminler de logaritmik ölçektedir. Ancak Gretl akıllı bir yazılımdır; forecast penceresinde genellikle veriyi otomatik olarak orijinal ölçeğe (anti-log alarak, yani $e^x$ işlemiyle) dönüştürür. Sonuç tablosundaki değerlerin binli rakamlar (gerçek yolcu sayıları) olduğunu kontrol edin. Eğer 4.5, 5.2 gibi küçük rakamlar görüyorsanız, bunlar logaritmik değerlerdir; üstelini almanız gerekir.

## ---

**Bölüm 3: Modül 2 \- CPU Anomali Tespiti (Anormallik Yakalama)**

Şimdi ekonometrik araç çantamızı alıp, çok farklı bir alana, IT operasyonlarına taşıyoruz. Senaryomuz şu: Bir bulut sunucusunun (örneğin AWS EC2) işlemci (CPU) kullanım verilerini izliyoruz. Amacımız, sistemin normal davranışının dışına çıktığı anları, yani anomalileri tespit etmek.

Bu tür analizler günümüzde **AIOps (Artificial Intelligence for IT Operations \- BT Operasyonları için Yapay Zeka)** başlığı altında büyük önem taşımaktadır.12

### **3.1. Veri Yapısı ve Zaman Damgası Sorunsalı**

Tipik bir sunucu performans verisi, CSV (Comma Separated Values \- Virgülle Ayrılmış Değerler) formatında gelir ve şuna benzer:

| Timestamp | CPU\_Usage | Memory\_Usage |
| :---- | :---- | :---- |
| 2025-10-01 08:00:00 | 12.5 | 45.2 |
| 2025-10-01 08:05:00 | 13.2 | 46.1 |
| ... | ... | ... |

Gretl'de çalışırken en sık karşılaşılan zorluk, bu tür CSV dosyalarındaki tarih/saat formatlarının doğru tanınmasıdır.13 Finansal veriler genellikle günlük veya aylıktır ve Gretl bunları kolayca tanır. Ancak "5 dakikalık" veya "saniyelik" IT verileri, özel bir işlem gerektirebilir.

**Veri İçe Aktarma İpuçları:**

1. **File** \> **Open data** \> **Import** \> **CSV...** menüsünden dosyanızı seçin.  
2. Gretl size "Bu veriyi zaman serisi olarak yorumlayayım mı?" diye sorduğunda "Evet" deyin.  
3. Eğer Gretl zaman sütununu tanıyamazsa, veriyi "Cross-sectional" (Kesit veri) olarak açacaktır. Bu durumda panik yapmayın.  
   * Veriyi açtıktan sonra **Data** \> **Dataset structure (Veri seti yapısı)** menüsüne gidin.16  
   * "Time series" seçeneğini seçin.  
   * Eğer veriniz düzenli aralıklıysa (örneğin her 5 dakikada bir), "Other" (Diğer) seçeneğini seçip, verinin sadece sıralı bir indeks (1, 2, 3...) olarak işlenmesini sağlayabilirsiniz. Anomali tespiti için mutlak tarihten ziyade, verinin sıralı olması ve örüntüsü daha önemlidir.

### **3.2. Anomali Nedir?**

Zaman serilerinde anomali (veya Outlier \- Aykırı Değer), beklenen modelden anlamlı derecede sapan gözlemlerdir.17 İki temel türe odaklanacağız:

1. **Point Anomaly (Noktasal Anomali):** Tek bir veri noktasının aniden fırlaması (Spike). Örneğin CPU'nun %15'ten aniden %95'e çıkıp bir sonraki adımda tekrar %15'e düşmesi.  
2. **Contextual Anomaly (Bağlamsal Anomali):** Değerin kendisi normal aralıkta olsa bile, o anki bağlamda anormal olması. Örneğin, CPU kullanımının gündüz %60 olması normaldir, ancak gece 03:00'te %60 olması bir anomalidir (crypto mining virüsü şüphesi?).

### **3.3. Yöntem: Kalıntı Analizi (Residual Analysis)**

Gretl'de "Tek Tıkla Anomali Bul" butonu yoktur. Ancak ekonometrik yaklaşım, "Isolation Forest" gibi kara kutu makine öğrenmesi algoritmalarından çok daha şeffaf ve güvenilirdir. Yöntemimiz şudur: **"Normali modelle, kalanı incele."**

Mantık basit: Eğer biz sistemin "normal" davranışını açıklayan bir model kurarsak, bu modelin hata yaptığı (açıklayamadığı) noktalar, tanım gereği anomalidir.

$$Kalıntı (Residual) \= Gerçek Değer (Actual) \- Tahmin Edilen Değer (Fitted)$$  
**Adım Adım Uygulama:**

1\. Normal Davranışı Modelleme (AR Modeli):  
CPU kullanımı genellikle güçlü bir otokorelasyona sahiptir; yani şu anki kullanım, 5 dakika önceki kullanımla ilişkilidir. Bu ilişkiyi yakalamak için basit bir AR(1) (Birinci Dereceden Otoregresif) model yeterlidir.

* **Model** \> **Ordinary Least Squares (En Küçük Kareler)** menüsünü açın.  
* Bağımlı değişken olarak cpu\_usage seçin.  
* Bağımsız değişkenler (Regressor) listesine cpu\_usage'ı eklemeyin. Bunun yerine, "lags" (gecikmeler) özelliğini kullanacağız. Gretl'de bir değişkenin gecikmesini eklemek için genellikle değişken listesinden seçip sağ tıklayarak **Add lags...** diyebilir veya model penceresinde manuel olarak cpu\_usage(-1) yazabilirsiniz.  
* Yani modelimiz: $CPU\_t \= \\alpha \+ \\beta \\cdot CPU\_{t-1} \+ u\_t$ olacaktır.  
* **OK** deyin.

2\. Kalıntıları (Residuals) Çıkarma:  
Model sonucunda elde ettiğimiz $u\_t$ (hata terimi), modelin açıklayamadığı kısımdır. Normal zamanlarda bu hata terimi sıfıra yakın küçük rastgele değerler olmalıdır. Anomalilerde ise bu değer fırlayacaktır.

* Model penceresinde **Save (Kaydet)** \> **Residuals (Kalıntılar)** seçeneğine tıklayın.7  
* Veri setine uhat adında yeni bir değişken eklenecektir.

3\. Eşik Değer (Threshold) Belirleme:  
Hangi hata değeri "çok büyük" kabul edilir? İstatistikte yaygın kullanılan 3-Sigma Kuralı (Empirical Rule) burada devreye girer. Bir veri normal dağılıyorsa, verilerin %99.7'si ortalamadan $\\pm 3$ standart sapma uzaklıkta yer alır.20 Bunun dışına çıkan %0.3'lük kısım, potansiyel anomalidir.

* Ana pencerede uhat değişkenini seçin.  
* **Variable** \> **Summary statistics (Özet istatistikler)** diyerek standart sapmasını ($SD$) öğrenin. Diyelim ki $SD \= 2.0$.  
* O halde Eşik Değerimiz: $3 \\times 2.0 \= 6.0$.  
* Eğer herhangi bir andaki hata (mutlak değer olarak) 6.0'dan büyükse, o an bir anomalidir.

4\. Anomalileri Görselleştirme:  
Gretl'de bu noktaları görselleştirmek için bir "Dummy Variable" (Kukla Değişken) oluşturabiliriz.

* Menüden **Add** \> **Define new variable (Yeni değişken tanımla)** seçeneğine girin.  
* Formül kutusuna şu komutu (hansl dili) yazın:  
  anomaly\_flag \= abs(uhat) \> (3 \* sd(uhat))  
* Bu komut, anomali olan anlarda 1, normal anlarda 0 değerini alan bir değişken yaratır.  
* Son olarak cpu\_usage ve anomaly\_flag değişkenlerini aynı grafikte (farklı eksenlerde veya ölçeklendirilmiş olarak) çizdirerek, sistemin hangi anlarda "kırmızı alarm" verdiğini görebilirsiniz.

## ---

**Sonuç ve İleri Okuma**

Bu kapsamlı rehberde, Gretl'in sunduğu güçlü araçları kullanarak iki temel veri bilimi problemini çözdük. AirPassengers örneği ile verinin dilinden anlamayı, trendleri ve mevsimselliği yönetmeyi ve belirsizliği (Fan Chart ile) görselleştirmeyi öğrendik. CPU örneği ile de, basit bir regresyon modelinin (OLS/AR) aslında nasıl güçlü bir anomali tespit motoruna dönüşebileceğini gördük.

Ekonometri ve veri analizi, sadece yazılım kullanmak değil, verinin arkasındaki hikayeyi okuyabilmektir. Gretl, bu hikayeyi okumanız için size temiz bir sayfa ve güçlü bir kalem sunar; hikayeyi yazmak ise sizin analitik düşünce yapınıza kalmıştır.

### **Özet Tablo: Kullanılan Temel Gretl Komutları ve İşlevleri**

Aşağıdaki tablo, ders boyunca kullandığımız menü adımlarını ve karşılık gelen teknik kavramları özetlemektedir.

| İşlem | Menü Yolu / Komut | Teknik Karşılığı (Kavram) |
| :---- | :---- | :---- |
| **Veri Yükleme** | File \> Open data \> Sample file | Data Ingestion |
| **Görselleştirme** | View \> Graph specified vars \> Time series plot | Exploratory Data Analysis (EDA) |
| **Dönüşüm** | Add \> Logs of selected variables | Variance Stabilization (Box-Cox) |
| **Model Kurma** | Model \> Time series \> ARIMA | Parametric Modeling |
| **Öngörü** | Analysis \> Forecasts | Forecasting (Out-of-Sample) |
| **Hata Kaydetme** | Save \> Residuals | Residual Diagnostics |
| **Değişken Yaratma** | Add \> Define new variable | Feature Engineering |

Unutmayın, en karmaşık model her zaman en iyi model değildir; veriyi en iyi açıklayan ve en az hatayı yapan model, en iyi modeldir (Parsimony Principle \- Tutumluluk İlkesi).

#### 

#### **Alıntılanan çalışmalar**

1. PRACTICAL ECONOMETRICS. I. REGRESSION MODELS PRAKTINĖ EKONOMETRIJA. I. REGRESINIAI MODELIAI KOMPUTERINĖS PRATYBOS, erişim tarihi Aralık 12, 2025, [http://www.statistika.mif.vu.lt/wp-content/uploads/2014/05/PE.I-2013-CompLabs.2013.12.17\_t.pdf](http://www.statistika.mif.vu.lt/wp-content/uploads/2014/05/PE.I-2013-CompLabs.2013.12.17_t.pdf)  
2. Using gretl for Principles of Econometrics, 3rd Edition Version 1.01, erişim tarihi Aralık 12, 2025, [http://www.kufel.torun.pl/prog/ebook\_gretl.pdf](http://www.kufel.torun.pl/prog/ebook_gretl.pdf)  
3. PRACTICAL ECONOMETRICS. II. TIME SERIES ANALYSIS COMPUTER LABS \*\*\* PRAKTINĖ EKONOMETRIJA. II. LAIKINĖS SEKOS PRATYBOS KOMPUTER, erişim tarihi Aralık 12, 2025, [http://www.statistika.mif.vu.lt/wp-content/uploads/2014/05/PE.II-CompLabs-2014.04.01\_t.pdf](http://www.statistika.mif.vu.lt/wp-content/uploads/2014/05/PE.II-CompLabs-2014.04.01_t.pdf)  
4. Seasonal Adjustment with the R Packages x12 and x12GUI \- ResearchGate, erişim tarihi Aralık 12, 2025, [https://www.researchgate.net/publication/261507877\_Seasonal\_Adjustment\_with\_the\_R\_Packages\_x12\_and\_x12GUI](https://www.researchgate.net/publication/261507877_Seasonal_Adjustment_with_the_R_Packages_x12_and_x12GUI)  
5. January 2024 \- Gretl-users \- gretlml.univpm.it \- List Index, erişim tarihi Aralık 12, 2025, [https://gretlml.univpm.it/hyperkitty/list/gretl-users@gretlml.univpm.it/2024/1/](https://gretlml.univpm.it/hyperkitty/list/gretl-users@gretlml.univpm.it/2024/1/)  
6. grmod Interface for gretl estimation commands \- RDocumentation, erişim tarihi Aralık 12, 2025, [https://www.rdocumentation.org/packages/Rgretl/versions/0.2.2/topics/grmod](https://www.rdocumentation.org/packages/Rgretl/versions/0.2.2/topics/grmod)  
7. Gretl Command Reference \- SourceForge, erişim tarihi Aralık 12, 2025, [https://gretl.sourceforge.net/gretl-help/cmdref.html](https://gretl.sourceforge.net/gretl-help/cmdref.html)  
8. Gretl Tutorial 6: Modeling and Forecasting Time Series Data \- YouTube, erişim tarihi Aralık 12, 2025, [https://www.youtube.com/watch?v=VTc9Ioy9bkQ](https://www.youtube.com/watch?v=VTc9Ioy9bkQ)  
9. gretl time series analysis \- YouTube, erişim tarihi Aralık 12, 2025, [https://www.youtube.com/watch?v=atkf4PHThDc](https://www.youtube.com/watch?v=atkf4PHThDc)  
10. List of statistics articles \- Wikipedia, erişim tarihi Aralık 12, 2025, [https://en.wikipedia.org/wiki/List\_of\_statistics\_articles](https://en.wikipedia.org/wiki/List_of_statistics_articles)  
11. Items where Subject is "C51 \- Model Construction and Estimation", erişim tarihi Aralık 12, 2025, [https://mpra.ub.uni-muenchen.de/view/subjects/C51.html](https://mpra.ub.uni-muenchen.de/view/subjects/C51.html)  
12. EC2 Instance Metrics(CPU,Memory and Disk Usage) \- Kaggle, erişim tarihi Aralık 12, 2025, [https://www.kaggle.com/datasets/sakthivelank/ec2-instance-metricscpumemory-and-disk-usage](https://www.kaggle.com/datasets/sakthivelank/ec2-instance-metricscpumemory-and-disk-usage)  
13. Re: \[Gretl-users\] csv date parsing \- UNIVPM, erişim tarihi Aralık 12, 2025, [https://gretlml.univpm.it/hyperkitty/list/gretl-users@gretlml.univpm.it/message/MCQFPSDFHMYUX35STAZUENSWZCLBEKEM/](https://gretlml.univpm.it/hyperkitty/list/gretl-users@gretlml.univpm.it/message/MCQFPSDFHMYUX35STAZUENSWZCLBEKEM/)  
14. Import data from csv with timestamps \- Stack Overflow, erişim tarihi Aralık 12, 2025, [https://stackoverflow.com/questions/32520768/import-data-from-csv-with-timestamps](https://stackoverflow.com/questions/32520768/import-data-from-csv-with-timestamps)  
15. 308 Date format issue when importing data into Gretl \- SourceForge, erişim tarihi Aralık 12, 2025, [https://sourceforge.net/p/gretl/bugs/308/](https://sourceforge.net/p/gretl/bugs/308/)  
16. Econometric Analysis Tool — Gretl | by Priyanka R \- Medium, erişim tarihi Aralık 12, 2025, [https://medium.com/@priramz11/econometric-analysis-tool-gretl-90ead518ecfb](https://medium.com/@priramz11/econometric-analysis-tool-gretl-90ead518ecfb)  
17. Mastering Anomaly Detection in Time Series Data: Techniques and Insights \- Medium, erişim tarihi Aralık 12, 2025, [https://medium.com/@ketan31kumar/mastering-anomaly-detection-in-time-series-data-techniques-and-insights-98fbe94c4258](https://medium.com/@ketan31kumar/mastering-anomaly-detection-in-time-series-data-techniques-and-insights-98fbe94c4258)  
18. Anomaly Detection in Time Series: A Comprehensive Evaluation \- VLDB Endowment, erişim tarihi Aralık 12, 2025, [http://vldb.org/pvldb/vol15/p1779-wenig.pdf](http://vldb.org/pvldb/vol15/p1779-wenig.pdf)  
19. OLS regression and detrending in GRETL \- Cross Validated \- Stack Exchange, erişim tarihi Aralık 12, 2025, [https://stats.stackexchange.com/questions/292658/ols-regression-and-detrending-in-gretl](https://stats.stackexchange.com/questions/292658/ols-regression-and-detrending-in-gretl)  
20. How to build a live time series anomaly detection model \- Metaplane, erişim tarihi Aralık 12, 2025, [https://www.metaplane.dev/blog/how-to-build-a-live-time-series-anomaly-detection-model](https://www.metaplane.dev/blog/how-to-build-a-live-time-series-anomaly-detection-model)