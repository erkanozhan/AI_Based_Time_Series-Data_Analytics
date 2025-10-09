
---

# Yapay Zeka Tabanlı Zaman Serisi ve Veri Analizi Ders Notları

---

### Giriş: Zaman Serisi Analizine Bakış

Zaman serisi, belirli bir zaman aralığında ardışık olarak gözlemlenen veri noktalarının bütünüdür. Örneğin, bir hastanedeki günlük hasta sayısı, bir şirketin aylık satış rakamları veya bir meteoroloji istasyonunda saatlik sıcaklık ölçümleri zaman serisi verilerine örnektir. Box ve Jenkins’e göre zaman serisi, “zamana bağlı olarak düzenli aralıklarla kaydedilen gözlemler dizisidir.” Zaman serisi analizi, geçmiş verilerden öğrenerek gelecekteki değerleri tahmin etmeye, anormallikleri tespit etmeye ve temel desenleri ortaya çıkarmaya olanak tanır.

Zaman serisi analizi finans (hisse senedi fiyatları, döviz kurları), ekonomi (enflasyon, işsizlik oranları), sağlık (hasta takibi, salgın analizi), enerji (elektrik tüketimi, üretim tahmini), mühendislik (sensör verileri, arıza tahmini) ve çevre bilimleri (hava durumu, iklim değişikliği) gibi birçok alanda kritik öneme sahiptir. Örneğin, elektrik talebinin saatlik değişimi enerji yönetiminde, kalp atış hızının zaman içindeki değişimi ise sağlıkta önemli bilgiler sunar.

Zaman serisi verilerinin analizi, aşağıdaki temel soruları yanıtlamayı amaçlar:
- Geçmişteki desenler nelerdir?
- Gelecekteki değerler nasıl tahmin edilebilir?
- Serideki olağan dışı değişimler nasıl tespit edilir?
- Zaman serisinin bileşenleri (trend, mevsimsellik, döngü, rastgelelik) nelerdir?

Bu nedenle zaman serisi analizi, veri bilimi ve yapay zeka uygulamalarında önemli bir rol oynar ve karar destek sistemlerinin temelini oluşturur.

---

### Zaman Serisi Temel Kavramları

Zaman serisi analizinin temelini oluşturan önemli kavramlar şunlardır:

-   **Gözlem (Observation):** $x_t$ ile gösterilir, $t$ zamanındaki veri noktasıdır.
    -   *Örnek:* Bir bankadaki günlük işlem sayısı, $x_{15}$ = 120 işlem.

    ```markdown
    Zaman (t):    1   2   3   4   5
    Gözlem (x_t): 80  95 120 110 130

    Grafik:
        130 |         ●
        120 |       ●
        110 |     ●
         95 |   ●
         80 | ●
                +---------------------
                    1   2   3   4   5   (Zaman)
    ```

-   **Zaman Dizisi (Time Index):** $t = 1, 2, ..., T$ şeklinde tanımlanır.
    -   *Örnek:* Bir yıl boyunca her ay ölçülen sıcaklık değerleri için $t = 1$ Ocak, $t = 12$ Aralık.

    ```markdown
    Zaman Dizisi:
    t:   1   2   3   ...  T
             │   │   │        │
             ●───●───●───...──●
    ```

-   **Trend:** Zaman serisindeki uzun vadeli artış veya azalış.
    -   *Örnek:* Bir e-ticaret sitesinin yıllık satışlarının sürekli artması.

    ```markdown
    Trend:
    Değer
        5 |        ●
        4 |      ●
        3 |    ●
        2 |  ●
        1 |●
            +-------------
             t →
    ```

-   **Mevsimsellik (Seasonality):** Belirli periyotlarda tekrar eden dalgalanmalar.
    -   *Örnek:* Yaz aylarında artan dondurma satışları, kışın azalan satışlar.

    ```markdown
    Mevsimsellik:
    Değer
        5 |   ●   ●   ●
        3 | ●   ●   ●
        1 |●___●___●___
             t →
    ```

-   **Durağanlık (Stationarity):** Serinin istatistiksel özelliklerinin zamanla değişmemesi.
    -   *Örnek:* Ortalama ve varyansı yıl boyunca sabit kalan sıcaklık ölçümleri.

    ```markdown
    Durağanlık:
    Değer
        5 | ● ● ● ● ● ●
        3 | ● ● ● ● ● ●
        1 |●_●_●_●_●_●_
             t →
    ```

Bu kavramlar, zaman serisi analizinin temelini oluşturur. Her bir şekil, ilgili kavramı görsel olarak daha anlaşılır ve sade biçimde temsil eder.

---

### Zaman Serisi Bileşenleri

Bir zaman serisi genellikle şu şekilde modellenir:
$$
x_t = T_t + S_t + C_t + I_t
$$
-   $T_t$: Trend bileşeni (uzun vadeli artış veya azalış)
-   $S_t$: Mevsimsellik bileşeni (belirli periyotlarda tekrar eden dalgalanmalar)
-   $C_t$: Döngüsel bileşen (iş döngüleri gibi uzun vadeli, periyodik olmayan dalgalanmalar)
-   $I_t$: Rastgele (iradi) bileşen (açıklanamayan, öngörülemeyen dalgalanmalar veya gürültü)

---

### Zaman Serisi Sınıflandırması ve Tipleri

Zaman serileri, ölçüm aralığı, değişken sayısı ve istatistiksel özellikleri gibi çeşitli kriterlere göre sınıflandırılabilir. Bu sınıflandırma, uygun analiz ve modelleme yönteminin seçilmesi için kritik öneme sahiptir.

1.  **Ölçüm Zamanına Göre Sınıflandırma:**
    *   **Kesikli Zaman Serisi (Discrete-Time Series):** Gözlemlerin belirli ve ayrık zaman noktalarında ($t_1, t_2, ..., t_n$) yapıldığı serilerdir. Finansal piyasalardaki günlük kapanış fiyatları, bir şirketin üç aylık gelirleri veya yıllık Gayri Safi Yurt İçi Hasıla (GSYİH) verileri bu kategoriye girer. Pratikte analiz edilen zaman serilerinin büyük çoğunluğu kesiklidir.
        *   **Şekil: Kesikli Zaman Serisi**
            ```mermaid
            graph LR
                T1(Zaman 1) -- Gözlem 1 --> D1(Değer 1);
                T2(Zaman 2) -- Gözlem 2 --> D2(Değer 2);
                T3(Zaman 3) -- Gözlem 3 --> D3(Değer 3);
            ```
    *   **Sürekli Zaman Serisi (Continuous-Time Series):** Gözlemlerin zamanın her anında mevcut olduğu teorik bir durumu ifade eder. Fiziksel sensörlerden alınan veriler (örneğin, bir sismografın kaydettiği yer hareketi) veya bir elektrokardiyogram (EKG) sinyali bu türe örnektir. Analiz için bu tür veriler genellikle yüksek frekansta örneklenerek kesikli hale getirilir.
        *   **Şekil: Sürekli Zaman Serisi (Kavramsal)**
        ```
        Zaman --->
        |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
        ^                                ^
        Başlangıç                      Bitiş
        (Veri akışı kesintisizdir)
        ```

2.  **Değişken Sayısına Göre Sınıflandırma:**
    *   **Tek Değişkenli (Univariate) Zaman Serisi:** Tek bir değişkenin zaman içindeki değişimini inceler. Örneğin, sadece bir hisse senedinin fiyatının veya bir şehrin aylık ortalama sıcaklığının analizi. Model, serinin kendi geçmiş değerlerine dayanır ($x_t = f(x_{t-1}, x_{t-2}, ...)$).
        *   **Şekil: Tek Değişkenli Seri**
            ```mermaid
            graph LR
                subgraph "Sıcaklık Değerleri"
                    S1(5°C) --> S2(7°C) --> S3(6°C) --> S4(8°C);
                end
            ```
    *   **Çok Değişkenli (Multivariate) Zaman Serisi:** İki veya daha fazla değişkenin aynı zaman diliminde eş zamanlı olarak gözlemlendiği serilerdir. Örneğin, bir ülkenin Gayri Safi Yurt İçi Hasıla (GSYİH), enflasyon oranı ve işsizlik oranlarının birlikte incelenmesi. Bu tür analizlerde değişkenler arasındaki nedensellik ve korelasyon ilişkileri (örneğin, Vektör Otoregresyon (VAR) modelleri ile) de modellenir.
        *   **Şekil: Çok Değişkenli Seri**
            ```mermaid
            graph TD
                subgraph Zaman Noktası: T1
                    A1("GSYİH: %2")
                    B1("Enflasyon: %5")
                end
                subgraph Zaman Noktası: T2
                    A2("GSYİH: %2.1")
                    B2("Enflasyon: %5.2")
                end
                T1 --> T2;
            ```

3.  **İstatistiksel Özelliklere Göre Sınıflandırma:**
    *   **Durağan (Stationary) Zaman Serisi:** Serinin istatistiksel özelliklerinin (ortalama, varyans, otokovaryans) zaman içinde değişmediği serilerdir. Yani, serinin herhangi bir zaman dilimindeki davranışları, başka bir zaman dilimindeki davranışlarına istatistiksel olarak benzerdir. Otoregresif Bütünleşik Hareketli Ortalama (ARIMA - Autoregressive Integrated Moving Average) gibi birçok klasik zaman serisi modeli, serinin durağan olmasını veya durağanlaştırılmasını gerektirir.
        *   **Şekil: Durağan Seri (Sabit Ortalama Etrafında Dalgalanma)**
        ```
        Değer
          ^
          |   ●   ●     ●
        --|--●-●---●-●---- (Ortalama)
          | ●     ●   ●
          +----------------> Zaman
        ```
    *   **Durağan Olmayan (Non-Stationary) Zaman Serisi:** Ortalama, varyans gibi istatistiksel özellikleri zamanla değişen serilerdir. Trend veya mevsimsellik içeren seriler genellikle durağan değildir. Örneğin, sürekli artış gösteren bir teknoloji şirketinin hisse senedi fiyatları durağan olmayan bir seridir. Analizden önce bu serilerin fark alma (differencing) gibi yöntemlerle durağanlaştırılması gerekebilir.
        *   **Şekil: Durağan Olmayan Seri (Artan Trend)**
        ```
        Değer
          ^
          |         ●
          |       ●
          |     ●
          |   ●
          | ●
          +----------------> Zaman
        ```

4.  **Rastgelelik Durumuna Göre Sınıflandırma:**
    *   **Deterministik Zaman Serisi:** Gelecekteki değerleri, geçmiş değerler kullanılarak hatasız bir şekilde tahmin edilebilen serilerdir. Bu seriler genellikle matematiksel bir fonksiyonla (örneğin, $x_t = \cos(2\pi ft)$) ifade edilebilir ve rastgele bir bileşen içermezler.
        *   **Şekil: Deterministik Süreç**
            ```mermaid
            graph LR
                A(Geçmiş Değerler) -- f(x) --> B(Gelecek Değer - Hatasız Tahmin);
            ```
    *   **Stokastik Zaman Serisi:** Gelecekteki değerleri belirsizlik içeren ve rastgele bir bileşene sahip olan serilerdir. Gerçek dünyadaki zaman serilerinin neredeyse tamamı stokastiktir. Bu seriler, geçmiş değerlere ek olarak bir hata terimi ($\epsilon_t$) içeren stokastik süreçlerle modellenir.
        *   **Şekil: Stokastik Süreç**
            ```mermaid
            graph LR
                A(Geçmiş Değerler) -- f(x) --> B(Tahmin);
                C(Rastgele Hata ε) --> D(Gerçek Gelecek Değer);
                B -- + --> D;
            ```

---

### R Programlama ile Tarih ve Zaman Nesneleri

Zaman serisi analizinde, verinin doğru zaman formatında olması kritik öneme sahiptir. R, bu tür verileri işlemek için güçlü araçlar sunar.

#### Tarih Formatı Sorunu

Tarih formatları ülkeden ülkeye değişebilir ve bu durum veri entegrasyonunda sorunlara yol açabilir. Örneğin, `01/02/2024` tarihi Amerika'da 2 Ocak 2024 anlamına gelirken, Avrupa'da 1 Şubat 2024'ü, Japonya'da ise 2024 Ocak 1'i ifade edebilir. Bu nedenle, veri alırken daima kaynağın tarih formatını kontrol etmek önemlidir.

#### R'da Tarih Nesneleri

R'da iki temel tarih/zaman sınıfı bulunur:
1.  **Date**: Sadece tarih bilgisini (gün, ay, yıl) tutar.
2.  **POSIXct/POSIXlt**: Tarih, saat ve saat dilimi bilgilerini içerir.

```r
# Bugünün tarihini al
bugun <- Sys.Date()
print(bugun)
class(bugun)  # "Date" çıktısı verir

# Şu anki zamanı al
simdi <- Sys.time()
print(simdi)
class(simdi)  # "POSIXct" "POSIXt" çıktısı verir
```

#### Tarih Oluşturma

ISO 8601 formatı (YYYY-MM-DD) evrensel kabul gördüğü için tarih oluşturmada tercih edilmelidir.

```r
# Doğru formatta tarih oluşturma
tarih1 <- as.Date("2024-03-15")
print(tarih1)

# Farklı formatlardaki tarihleri okumak için 'format' parametresi kullanılır
# Amerikan formatı (MM/DD/YYYY)
tarih_us <- as.Date("03/15/2024", format = "%m/%d/%Y")
print(tarih_us)

# Avrupa formatı (DD/MM/YYYY)
tarih_eu <- as.Date("15/03/2024", format = "%d/%m/%Y")
print(tarih_eu)

# Uzun format
tarih_uzun <- as.Date("15 Mart 2024", format = "%d %B %Y")
print(tarih_uzun)
```

**Sık Kullanılan Format Kodları:**
-   `%Y`: 4 haneli yıl (örn: 2024)
-   `%y`: 2 haneli yıl (örn: 24)
-   `%m`: Ay numarası (01-12)
-   `%B`: Ay ismi (örn: Ocak, Şubat)
-   `%b`: Kısa ay ismi (örn: Oca, Şub)
-   `%d`: Gün (01-31)
-   `%A`: Gün ismi (örn: Pazartesi)
-   `%a`: Kısa gün ismi (örn: Pzt)

#### Gerçek Hayat Örneği: Excel Tarihleri

Excel'den gelen tarihler genellikle sayısal (numeric) formatta olabilir. R'a çevirirken Excel'in başlangıç tarihi (`1899-12-30`) dikkate alınmalıdır, çünkü R'ın başlangıç tarihi `1970-01-01`'dir.

```r
# Excel'den gelen sayısal tarih
excel_tarih <- 44927  # Excel'de 2024-01-01'e karşılık gelir

# R tarihine çevirme
r_tarih <- as.Date(excel_tarih, origin = "1899-12-30")
print(r_tarih)
```

#### `lubridate` Paketi

`lubridate` paketi, R'daki tarih ve zaman işlemlerini büyük ölçüde basitleştirir.

```r
library(lubridate)

# Farklı formatları otomatik tanır
tarih1 <- ymd("2024-03-15")      # Year-Month-Day
tarih2 <- dmy("15-03-2024")      # Day-Month-Year
tarih3 <- mdy("03/15/2024")      # Month-Day-Year

# Tarihten bilgi çekme
year(tarih1)   # 2024
month(tarih1)  # 3
day(tarih1)    # 15
wday(tarih1)   # Haftanın günü (1=Pazar, 7=Cumartesi)

# İsimli çıktı
month(tarih1, label = TRUE)  # "Mar"
wday(tarih1, label = TRUE)   # "Fri"
```

#### Tarih Aritmetiği

Tarihler üzerinde ekleme, çıkarma gibi aritmetik işlemler yapılabilir.

```r
# Tarihe gün ekleme
tarih <- as.Date("2024-01-01")
tarih + 30        # 30 gün sonra
tarih + 365       # 1 yıl sonra

# İki tarih arası fark
baslangic <- as.Date("2024-01-01")
bitis <- as.Date("2024-12-31")
fark <- bitis - baslangic
as.numeric(fark)  # 365 gün

# lubridate ile daha esnek tarih aritmetiği
baslangic + days(30)
baslangic + months(3)
baslangic + years(1)
```

#### Zaman Serisi İçin Tarih Dizisi Oluşturma

Zaman serisi analizinde belirli aralıklarda tarih dizileri oluşturmak sıkça kullanılır.

```r
# Günlük tarih dizisi
gunluk <- seq.Date(from = as.Date("2024-01-01"),
                   to = as.Date("2024-12-31"),
                   by = "day")
length(gunluk)  # 366 (2024 artık yıl)

# Aylık tarih dizisi
aylik <- seq.Date(from = as.Date("2024-01-01"),
                  to = as.Date("2024-12-31"),
                  by = "month")
length(aylik)  # 12

# İş günleri (hafta sonları hariç)
# library(bizdays) # bizdays paketi eklenmediyse yükleyin
# is_gunleri <- seq.Date(as.Date("2024-01-01"),
#                        as.Date("2024-12-31"),
#                        by = "day")
# is_gunleri <- is_gunleri[!wday(is_gunleri) %in% c(1,7)]
```

---

### R'da Zaman Serisi Nesnesi (`ts`)

R'da zaman serisi verilerini temsil etmek için `ts` (time series) sınıfı kullanılır.

#### `ts` Nesnesi Nedir?

`ts` nesnesi iki ana bileşenden oluşur:
1.  **Veri:** Sayısal değerler.
2.  **Zaman Bilgisi:** Serinin başlangıç zamanı (`start`), bitiş zamanı (`end`) ve gözlemlerin bir döngüde ne sıklıkla tekrarlandığını belirten frekansı (`frequency`).

```r
# Basit bir ts nesnesi oluşturma örneği
veri <- c(100, 105, 98, 112, 108, 115, 120, 118, 125, 130, 128, 135)

# Aylık veri, 2024 Ocak'tan başlıyor
satis_ts <- ts(data = veri,
               start = c(2024, 1),
               frequency = 12)

print(satis_ts)
```

**Yukarıdaki kodun çıktısı:**
```
     Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
2024 100  105   98  112  108  115  120  118  125  130  128  135
```

#### Frekans Kavramı

Frekans, bir zaman serisindeki döngünün tamamlanması için gereken gözlem sayısını ifade eder ve zaman serisi analizinde çok önemlidir:
-   Aylık veri: `frequency = 12`
-   Çeyreklik veri: `frequency = 4`
-   Yıllık veri: `frequency = 1`
-   Günlük veri: `frequency = 365` (veya artık yıl için 365.25)
-   Haftalık veri: `frequency = 52`
-   Saatlik veri (günlük döngü): `frequency = 24`

**Gerçek Veri Örneği:**

```r
# TSstudio paketinden gerçek veri (paketi yüklemediyseniz install.packages("TSstudio"))
library(TSstudio)
data(USgas)

# Veriyi incele
ts_info(USgas)
```

**Yukarıdaki kodun çıktısı (örnek):**
```
The USgas series is a ts object with 1 variable and 227 observations
Frequency: 12
Start time: 2000 1
End time: 2018 11
```
Bu çıktı, 227 aylık gözlem içeren, Ocak 2000'de başlayıp Kasım 2018'de biten aylık bir zaman serisi olduğunu gösterir.

#### `ts` Nesnesi Özellikleri

`ts` nesnesinin temel özelliklerine erişmek için çeşitli fonksiyonlar bulunur:

```r
data(USgas) # USgas veri setini tekrar yükleyelim

# Başlangıç ve bitiş zamanı
start(USgas)   # [1] 2000    1
end(USgas)     # [1] 2018   11

# Frekans
frequency(USgas)  # [1] 12

# Gözlemler arasındaki zaman aralığı (1/frekans)
deltat(USgas)     # [1] 0.08333333 (1/12)

# Döngü içindeki konum (örneğin, aylık veri için ay numarası)
head(cycle(USgas))  # [1] 1 2 3 4 5 6 (ilk 6 ay)

# Serinin uzunluğu (gözlem sayısı)
length(USgas)  # [1] 227
```

#### Veri Çekme ve Kesme

`window` fonksiyonu, zaman serisinin belirli bir alt kümesini almak için en sık kullanılan yöntemdir.

```r
# 2010-2015 yılları arasındaki veriyi seç
subset1 <- window(USgas,
                  start = c(2010, 1),
                  end = c(2015, 12))
print(subset1)

# Sadece Ocak aylarını seç (frekansı 1 olarak ayarlayarak)
ocaklar <- window(USgas,
                  start = c(2000, 1), # Başlangıç yılı ve ayı önemli
                  frequency = 1) # Her Ocak ayını ayrı bir seri gibi ele alır
# print(ocaklar) # Bu gösterim her Ocak ayını yeni bir "yıl" gibi gösterir

# Serinin son 24 gözlemini (son 2 yılını) al
son24 <- tail(USgas, 24)

# Serinin ilk 24 gözlemini (ilk 2 yılını) al
ilk24 <- head(USgas, 24)
```

#### Kendi `ts` Nesnenizi Oluşturma

Gerçek hayatta, veriler genellikle bir `data.frame` içinde bulunur ve `ts` nesnesine dönüştürülmesi gerekir.

```r
# Excel'den geldiği varsayılan veriler
tarihler <- seq.Date(as.Date("2020-01-01"),
                     by = "month",
                     length.out = 48) # 4 yıl, 48 ay
satislar <- rnorm(48, mean = 1000, sd = 100) # Rastgele satış değerleri

# data.frame oluştur
df <- data.frame(tarih = tarihler, satis = satislar)

# ts nesnesine çevirmek için başlangıç yılı ve ay bilgisi gerekir
library(lubridate) # lubridate paketi kullanılarak tarih bilgisi kolayca alınır
baslangic_yil <- year(min(df$tarih))
baslangic_ay <- month(min(df$tarih))

satis_ts <- ts(df$satis,
               start = c(baslangic_yil, baslangic_ay),
               frequency = 12) # Aylık veri olduğu için frekans 12

# Kontrol et
ts_info(satis_ts)
```

#### Temel Görselleştirme

`ts` nesneleri, `plot()` fonksiyonu ile kolayca görselleştirilebilir.

```r
# Temel plot
plot(USgas,
     main = "ABD Doğal Gaz Tüketimi",
     ylab = "Milyar Kübik Fit",
     xlab = "Yıl")

# TSstudio paketi ile interaktif görselleştirme
# library(TSstudio) # Yüklü değilse install.packages("TSstudio")
# ts_plot(USgas,
#         title = "ABD Doğal Gaz Tüketimi",
#         Ytitle = "Milyar Kübik Fit",
#         Xtitle = "Yıl")
```

---

### Zaman Serisi Analizinde Görselleştirme

Görselleştirme, veri içindeki desenleri, anormallikleri, trendleri ve mevsimselliği insan gözüyle hızlıca tespit etmenin en etkili yoludur.

-   **Zaman Serisi Grafiği:** Verinin zamana karşı çizilmesi, serinin genel yapısını gösterir. Örneğin, bir şehrin saatlik elektrik tüketim grafiği, gün içindeki (sabah ve akşam) ve hafta sonundaki talep düşüşlerini net bir şekilde gösterir.

    **Python ile Airlines Veri Seti Görselleştirme Örneği:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Airlines veri setini yükle (örnek: 'AirPassengers.csv' dosyasının yerel sistemde olduğu varsayılır)
    # Eğer dosya yoksa, aşağıdaki örnek veri ile devam edilebilir:
    # data = {'Month': pd.to_datetime(['1949-01-01', '1949-02-01', '1949-03-01', '1949-04-01']),
    #         'Passengers': [112, 118, 132, 129]}
    # df = pd.DataFrame(data)

    df = pd.read_csv('AirPassengers.csv')
    # 'Month' sütununu datetime tipine çevir
    df['Month'] = pd.to_datetime(df['Month'])
    # Zaman serisi grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(df['Month'], df['Passengers'], marker='o')
    plt.title('Aylık Yolcu Sayısı (Airlines Veri Seti)')
    plt.xlabel('Ay')
    plt.ylabel('Yolcu Sayısı')
    plt.grid(True)
    plt.show()
    ```

    **R ile Airlines Veri Seti Görselleştirme Örneği:**
    ```r
    # AirPassengers veri seti R'da gömülüdür
    data("AirPassengers")
    # Zaman serisi grafiği
    plot(AirPassengers,
         main = "Aylık Yolcu Sayısı (Airlines Veri Seti)",
         xlab = "Yıl",
         ylab = "Yolcu Sayısı",
         col = "blue",
         lwd = 2)
    grid()
    ```

    Python'da pandas ve matplotlib, R'da ise gömülü AirPassengers veri seti ve plot fonksiyonu kullanılmıştır.

    **Zaman Serisi Formatına Dönüştürülmüş Veri Örneği:** Her veri başlangıçta uygun formatta olmayabilir. Örneğin, ham işlem kayıtları (transaction logs) genellikle zaman serisi formatında değildir. Ancak bu veriler, belirli bir zaman aralığına (örneğin günlük) göre gruplanarak zaman serisi haline getirilebilir.

    **Python ile Ham Veriden Zaman Serisi Oluşturma ve Görselleştirme:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Örnek ham veri: işlem kayıtları
    data = {
        'islem_zamani': [
            '2024-06-01 10:15', '2024-06-01 12:30', '2024-06-02 09:45',
            '2024-06-02 14:10', '2024-06-03 11:00', '2024-06-03 16:20'
        ],
        'musteri_id': [101, 102, 103, 104, 105, 106],
        'urun': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    df = pd.DataFrame(data)
    df['islem_zamani'] = pd.to_datetime(df['islem_zamani'])

    # Günlük işlem sayısını grupla
    daily_counts = df.groupby(df['islem_zamani'].dt.date).size()

    # Zaman serisi grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(daily_counts.index, daily_counts.values, marker='o')
    plt.title('Günlük İşlem Sayısı (Ham Veriden Zaman Serisi)')
    plt.xlabel('Tarih')
    plt.ylabel('İşlem Sayısı')
    plt.grid(True)
    plt.show()
    ```
    **R ile Ham Veriden Zaman Serisi Oluşturma ve Görselleştirme:**
    ```r
    # Örnek ham veri: işlem kayıtları
    data <- data.frame(
        islem_zamani = c(
            "2024-06-01 10:15", "2024-06-01 12:30", "2024-06-02 09:45",
            "2024-06-02 14:10", "2024-06-03 11:00", "2024-06-03 16:20"
        ),
        musteri_id = c(101, 102, 103, 104, 105, 106),
        urun = c("A", "B", "A", "C", "B", "A")
    )
    data$islem_zamani <- as.POSIXct(data$islem_zamani)

    # Günlük işlem sayısını grupla
    library(dplyr)
    library(ggplot2)
    daily_counts <- data %>%
        mutate(tarih = as.Date(islem_zamani)) %>%
        group_by(tarih) %>%
        summarise(islem_sayisi = n())

    # Zaman serisi grafiği
    ggplot(daily_counts, aes(x = tarih, y = islem_sayisi)) +
        geom_line() +
        geom_point() +
        labs(title = "Günlük İşlem Sayısı (Ham Veriden Zaman Serisi)",
                 x = "Tarih", y = "İşlem Sayısı") +
        theme_minimal()
    ```
    Bu örnekte, ham işlem kayıtları günlük bazda gruplanarak zaman serisi formatına dönüştürülmüş ve görselleştirilmiştir.
    -   **ACF (Autocorrelation Function - Otokorelasyon Fonksiyonu) ve PACF (Partial Autocorrelation Function - Kısmi Otokorelasyon Fonksiyonu) Grafikleri:** Model belirleme aşamasında kritik öneme sahiptir.

        -   **ACF Notasyonu:** $\rho_k$ ile gösterilir ve bir zaman serisinin kendi gecikmeli (lagged) değerleriyle olan korelasyonunu ölçer. ACF grafiğinde çubukların yavaşça azalması trendin varlığına, belirli aralıklarda tekrar eden yüksek çubuklar ise mevsimselliğe işaret eder.

            *   **Formül:**
                $$
                \rho_k = \frac{\text{Cov}(x_t, x_{t-k})}{\text{Var}(x_t)} = \frac{\sum_{t=k+1}^{T} (x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{T} (x_t - \bar{x})^2}
                $$

            *   **Python ile ACF Hesabı:**
                ```python
                from statsmodels.tsa.stattools import acf
                import numpy as np

                # Örnek veri
                data = np.array([20, 22, 21, 23, 24])
                
                # ACF hesapla
                acf_values = acf(data, nlags=2)
                print("Lag-1 ACF:", acf_values[1])
                ```

            *   **R ile ACF Hesabı:**
                ```r
                # Örnek veri
                data <- c(20, 22, 21, 23, 24)
                
                # ACF hesapla
                acf_result <- acf(data, plot = FALSE)
                print(paste("Lag-1 ACF:", acf_result$acf[2]))
                ```  *   **Formül:**
            $$
            \rho_k = \frac{\text{Cov}(x_t, x_{t-k})}{\text{Var}(x_t)} = \frac{\sum_{t=k+1}^{T} (x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{T} (x_t - \bar{x})^2}
            $$
        *   **Örnek Hesaplama (ACF Lag-1):**
            Diyelim ki elimizde 5 günlük sıcaklık verisi var: $x =$
            1. Ortalama: $\bar{x} = (20 + 22 + 21 + 23 + 24)/5 = 22$
            2. Pay (kovaryans):
                - $(22-22)(20-22) = 0 \times -2 = 0$
                - $(21-22)(22-22) = -1 \times 0 = 0$
                - $(23-22)(21-22) = 1 \times -1 = -1$
                - $(24-22)(23-22) = 2 \times 1 = 2$
                - Toplam pay: $0 + 0 + (-1) + 2 = 1$
            3. Payda (varyans):
                - $(20-22)^2 + (22-22)^2 + (21-22)^2 + (23-22)^2 + (24-22)^2 = (-2)^2 + 0^2 + (-1)^2 + 1^2 + 2^2 = 4 + 0 + 1 + 1 + 4 = 10$
            4. $\rho_1 = 1 / 10 = 0.1$


ACF Yorumlama:
    -   **PACF Notasyonu:** $\phi_{kk}$ ile gösterilir ve iki zaman noktası arasındaki doğrudan ilişkiyi, aradaki diğer gecikmelerin etkisini ortadan kaldırarak ölçer. PACF grafiğinde, belirli bir gecikme için çubuğun yüksek olması, o gecikmenin seride doğrudan etkili olduğunu gösterir. ACF ve PACF, Box-Jenkins metodolojisinde ARIMA modellerinin mertebelerini (p, q) belirlemek için birlikte kullanılır (Box & Jenkins, 1970).
     
    ** PACF (Kısmi Otokorelasyon) Kavramsal Gösterimi:**
                    ```mermaid
                    graph TD
                        subgraph "PACF Hesabı"
                        direction LR
                        X_t_2["x_t-2"] -->|"Doğrudan İlişki"| X_t["x_t"]
                        X_t_1["x_t-1"] -.->|"Dolaylı Etki"| X_t
                        X_t_2 -.->|"Dolaylı Etki"| X_t_1
                        end
                    ```
          
                *   **PACF'nin Önemi:**
                    -   İki zaman noktası arasındaki doğrudan ilişkiyi ölçer
                    -   Aradaki noktaların etkisini filtreler
                    -   ARIMA modellerinde AR teriminin derecesini belirlemede kullanılır
          
                *   **PACF Hesaplama:**
                    1.  Lag-1 için: $$x_t = \phi_{11}x_{t-1} + \epsilon_t$$
                    2.  Lag-2 için: $$x_t = \phi_{21}x_{t-1} + \phi_{22}x_{t-2} + \epsilon_t$$
                    3.  Lag-k için: $$x_t = \sum_{i=1}^k \phi_{ki}x_{t-i} + \epsilon_t$$
          
                *   **Kod Örnekleri:**
                    ```python
                    # Python ile PACF
                    from statsmodels.tsa.stattools import pacf
                    import numpy as np
                    
                    data = np.array([20, 22, 21, 23, 24])
                    pacf_values = pacf(data, nlags=2)
                    print(f"Lag-2 PACF: {pacf_values[2]:.3f}")
                    ```
          
                    ```r
                    # R ile PACF
                    data <- c(20, 22, 21, 23, 24)
                    pacf_result <- pacf(data, plot = FALSE)
                    cat("Lag-2 PACF:", round(pacf_result$acf[2], 3))
                    ```
          
            -   **Mevsimsel Ayrıştırma Grafiği (Seasonal Decomposition Plot):** Seriyi bileşenlerine ayırarak görselleştirir: Trend, Mevsimsellik ve Artıklar (Rastgele Gürültü). Bu, bir e-ticaret sitesinin satış verilerinde uzun vadeli büyüme trendini, tatil dönemlerindeki mevsimsel artışlardan ve beklenmedik satış dalgalanmalarından ayırmayı sağlar.
          
                ```mermaid
                graph TD
                    subgraph "Zaman Serisi Ayrıştırması"
                    A[Orijinal Seri: x_t] --> B[Trend: T_t];
                    A --> C[Mevsimsellik: S_t];
                    A --> D[Artıklar/Gürültü: I_t];
                    end
                ```

### R'da Zaman Serisi Veri Manipülasyonu ve Ayrıştırma

R, zaman serisi verilerini işlemek, dönüştürmek ve ayrıştırmak için güçlü fonksiyonlar sunar.

#### `aggregate()` - Zaman Serisini Toparlama

Aylık, günlük gibi yüksek frekanslı verileri daha düşük frekanslı verilere (çeyreklik, yıllık) dönüştürmek için `aggregate()` fonksiyonu kullanılır.

```r
data(USgas)

# Aylık veriyi yıllığa çevir ve toplamını al
USgas_yillik <- aggregate(USgas,
                          nfrequency = 1,
                          FUN = sum)
print(USgas_yillik)

# Aylık veriyi çeyrekliğe çevir ve toplamını al
USgas_ceyrek <- aggregate(USgas,
                          nfrequency = 4,
                          FUN = sum)
print(USgas_ceyrek)
```
**Not:** `FUN` parametresi `mean`, `median`, `max` gibi farklı fonksiyonlar için değiştirilebilir.

#### `lag()` - Gecikmeli Değerler

Zaman serisi analizinde bir önceki zaman dilimindeki değerler (gecikmeli değerler - lagged values) önemli bir rol oynar. `stats::lag()` fonksiyonu bu değerleri hesaplamak için kullanılır.

```r
# Birinci lag (1 ay önceki değer)
USgas_lag1 <- stats::lag(USgas, k = -1)

# 12. lag (1 yıl önceki değer, aylık veri için)
USgas_lag12 <- stats::lag(USgas, k = -12)

# Orijinal seri ile gecikmeli değerleri karşılaştırma
head(cbind(USgas, USgas_lag12), 15)
```
**Not:** `k` parametresi negatif olduğunda geçmiş değerleri, pozitif olduğunda ise gelecek değerleri ifade eder.

#### `decompose()` - Zaman Serisini Ayrıştırma

`decompose()` fonksiyonu, bir zaman serisini trend, mevsimsellik ve rastgele bileşenlerine ayırmak için kullanılır.

```r
# USgas serisini ayrıştır
USgas_ayristir <- decompose(USgas)

# Ayrıştırılmış bileşenleri görselleştir
plot(USgas_ayristir)
```
Bu işlem, serinin altında yatan yapıları anlamak için dört grafik gösterir: orijinal seri, trend bileşeni, mevsimsel bileşen ve rastgele bileşen.

```r
# Ayrıştırılmış bileşenlere ayrı ayrı erişim
trend <- USgas_ayristir$trend
sezonluk <- USgas_ayristir$seasonal
rastgele <- USgas_ayristir$random

# İlk ve son değerler hareketli ortalama hesaplaması nedeniyle NA (Not Available) olabilir
head(trend, 15)
```

#### `dplyr` ile Veri İşleme

Gerçek dünya verileri genellikle `data.frame` formatında gelir ve zaman serisi analizi öncesinde düzenleme gerektirebilir. `dplyr` ve `lubridate` paketleri bu konuda oldukça yardımcıdır.

```r
library(dplyr)
library(lubridate)

# Düzensiz, günlük satış verisi oluşturma
df <- data.frame(
  tarih = seq.Date(as.Date("2023-01-01"), by = "day", length.out = 365),
  satis = rnorm(365, 1000, 100)
)

# Günlük veriyi aylık ortalama satışa dönüştürme
aylik_ort <- df %>%
  mutate(ay = floor_date(tarih, "month")) %>% # Her tarihi ayın başına yuvarla
  group_by(ay) %>% # Ay bazında grupla
  summarise(
    ortalama_satis = mean(satis),
    toplam_satis = sum(satis),
    min_satis = min(satis),
    max_satis = max(satis),
    gun_sayisi = n()
  )

print(aylik_ort)
```

#### Gelişmiş Görselleştirme (`ggplot2`)

`ggplot2` paketi, R'da profesyonel ve özelleştirilebilir zaman serisi grafikleri oluşturmak için kullanılır. `ts` nesnelerini `ggplot2` ile kullanmak için önce `data.frame` formatına çevirmek gerekir.

```r
library(ggplot2)

# USgas ts nesnesini data.frame'e dönüştür
df_gg <- data.frame(
  tarih = as.Date(time(USgas)), # Zaman indeksini tarihe çevir
  deger = as.numeric(USgas)     # ts değerlerini sayısal vektöre çevir
)

# Profesyonel bir zaman serisi grafiği oluştur
ggplot(df_gg, aes(x = tarih, y = deger)) +
  geom_line(color = "blue", size = 0.8) +
  geom_smooth(method = "loess", color = "red", se = FALSE, linetype = "dashed") + # Yerel regresyon ile trend çizgisi ekle
  labs(title = "ABD Doğal Gaz Tüketimi (2000-2018)",
       subtitle = "ggplot2 ile Gelişmiş Görselleştirme",
       x = "Tarih",
       y = "Milyar Kübik Fit") +
  theme_minimal() + # Minimal tema kullan
  theme(plot.title = element_text(hjust = 0.5), # Başlığı ortaya hizala
        plot.subtitle = element_text(hjust = 0.5))
```

---

### Zaman Serisi Analizinde Kullanılan Yöntemler ve Modeller

Zaman serisi analizi, geçmiş verilerdeki kalıpları anlayarak geleceği öngörmeyi amaçlayan istatistiksel ve makine öğrenmesi yöntemlerini kapsar. Yöntem seçimi, verinin yapısına ve analizin amacına bağlıdır.

1.  **Keşifsel Analiz ve Temel İstatistikler**
    *   **Otokorelasyon Fonksiyonu (ACF):** Bir zaman serisinin, kendi geçmiş (gecikmeli) değerleriyle ne kadar ilişkili olduğunu ölçer. Örneğin, dondurma satışlarının bugünkü değeri, dünkü değeriyle yüksek korelasyon gösterebilir (lag-1). Aynı zamanda, 7 gün önceki değeriyle de (haftalık mevsimsellik nedeniyle) yüksek korelasyon gösterebilir (lag-7). ACF grafiği, bu tür periyodik desenleri ve mevsimselliği tespit etmek için kritik bir araçtır. Daha detaylı bilgi ve örnek hesaplamalar için "Zaman Serisi Analizinde Görselleştirme" bölümüne bakınız.
    *   **Kısmi Otokorelasyon Fonksiyonu (PACF):** İki zaman noktası arasındaki doğrudan ilişkiyi, aralarındaki gecikmelerin etkisini ortadan kaldırarak ölçer. Örneğin, $x_t$ ve $x_{t-2}$ arasındaki PACF, $x_{t-1}$'in aracı etkisini kaldırarak bu iki nokta arasındaki net korelasyonu gösterir. ACF ve PACF, Box-Jenkins metodolojisinde ARIMA modellerinin mertebelerini (p, q) belirlemek için birlikte kullanılır (Box & Jenkins, 1970). Daha detaylı bilgi ve örnek hesaplamalar için "Zaman Serisi Analizinde Görselleştirme" bölümüne bakınız.

2.  **Klasik İstatistiksel Modeller**
    *   **Hareketli Ortalama (Moving Average - MA):** Serideki rastgele gürültüyü veya kısa vadeli dalgalanmaları yumuşatarak ana eğilimi (trendi) daha net görmeyi sağlar. Örneğin, bir hisse senedinin 30 günlük hareketli ortalaması, günlük fiyat dalgalanmalarını filtreleyerek yatırımcının uzun vadeli yönelimi anlamasına yardımcı olur.
        *   **Not:** Basit Hareketli Ortalama (SMA) tüm noktalara eşit ağırlık verirken, **Üstel Ağırlıklı Hareketli Ortalama (Exponential Moving Average - EMA)** yakın geçmişteki verilere daha fazla ağırlık vererek yeni bilgilere daha hızlı adapte olur. Bu özellik, EMA'yı finansal piyasalar gibi dinamik sistemlerin analizinde daha popüler kılar.
            $$
            \text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
            $$
    *   **ARIMA (Autoregressive Integrated Moving Average) Modeli:** Durağan olmayan zaman serilerini modellemek için en yaygın kullanılan istatistiksel yöntemlerden biridir.
        *   **AR (Autoregressive - p):** Model, değişkenin geçmiş değerlerine bağlıdır. Örnek: Bir sonraki günün sıcaklığı, önceki günlerin sıcaklıklarına bağlıdır.
        *   **I (Integrated - d):** Seriyi durağan hale getirmek için fark alma (differencing) işlemi sayısıdır. Örnek: Sürekli artan GSYİH verisinin yıllık artış oranına dönüştürülmesi.
        *   **MA (Moving Average - q):** Model, geçmiş tahmin hatalarına bağlıdır. Örnek: Tahmin modelinin dünkü hatası, bugünkü tahmini düzeltmek için kullanılır.
        *   **Uygulama:** Bir perakende şirketinin aylık satışlarını tahmin etmek için SARIMA (Mevsimsel ARIMA) modeli kullanılabilir. Bu model, hem genel satış trendini hem de tatil dönemleri gibi yıllık tekrarlayan mevsimsel zirveleri modelleyebilir. (Hyndman & Athanasopoulos, 2018).

        ```mermaid
        graph TD
            A["Data Collection & Visualization"] --> B["Is Series Stationary?"]
            B -- "No" --> C["Differencing"]
            C --> D["ACF/PACF Model Selection (p,q)"]
            B -- "Yes" --> D
            D --> E["Estimate Model Parameters"]
            E --> F["Diagnostic Checks (Residuals White Noise?)"]
            F -- "Yes" --> G["Forecasting"]
            F -- "No" --> D
            subgraph BoxJenkinsMethodology
                direction LR
                D -- "p,d,q" --> E
                E -- "Model" --> F
            end
        ```

---

### Yapay Zeka ile Zaman Serisi Analizi

Klasik istatistiksel modellerin varsayımlarını karşılamayan karmaşık ve doğrusal olmayan ilişkileri modellemek için yapay zeka yöntemleri giderek daha fazla kullanılmaktadır.

-   **Makine Öğrenmesi (Machine Learning):**
    *   **Yaklaşım:** Zaman serisi problemi, denetimli bir öğrenme problemine dönüştürülür. Geçmiş değerler (örn: $x_{t-1}, x_{t-2}, ...$) girdi (özellik), gelecekteki değer ($x_t$) ise çıktı (hedef) olarak kullanılır.
    *   **Örnek:** Bir mağazanın günlük müşteri sayısını tahmin etmek için Gradient Boosting veya Random Forest gibi modeller kullanılabilir. Girdi özellikleri olarak geçmiş müşteri sayıları (lag features), haftanın günü, ay, tatil olup olmadığı gibi takvim özellikleri ve promosyon bilgileri verilebilir.
-   **Derin Öğrenme (Deep Learning):**
    *   **LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit):** Bu tekrarlayan sinir ağı (RNN) mimarileri, zaman serilerindeki uzun vadeli bağımlılıkları öğrenmek için tasarlanmıştır. Standart RNN'lerin karşılaştığı "kaybolan gradyan" (vanishing gradient) sorununu, "kapı" (gate) mekanizmaları sayesinde aşarlar (Hochreiter & Schmidhuber, 1997).
    *   **Uygulama:** Bir enerji santralinin bir sonraki saatteki üretimini tahmin etmek için LSTM modeli kullanılabilir. Model, geçmiş üretim verileri, hava durumu tahminleri (rüzgar hızı, güneşlenme) ve talep verileri gibi çok değişkenli serileri işleyerek karmaşık ilişkileri öğrenebilir.

    **LSTM Hücresinin Kavramsal Çalışması:**
    ```mermaid
    graph TD
        subgraph "LSTM Hücresi"
            direction LR
            C_prev[Önceki Hücre Durumu c_t-1] --> ForgetGate{Unutma Kapısı};
            Input[Girdi x_t] --> ForgetGate;
            H_prev[Önceki Gizli Durum h_t-1] --> ForgetGate;

            Input --> InputGate{Giriş Kapısı};
            H_prev --> InputGate;

            ForgetGate -- Karar: Neyi Unut? --> CellStateUpdate(Hücre Durumunu Güncelle);
            InputGate -- Karar: Neyi Ekle? --> CellStateUpdate;
            CellStateUpdate --> C_next[Yeni Hücre Durumu c_t];

            C_next --> OutputGate{Çıkış Kapısı};
            Input --> OutputGate;
            H_prev --> OutputGate;
            OutputGate -- Karar: Neyi Çıktı Ver? --> H_next[Yeni Gizli Durum h_t];
        end
        C_prev --> C_next;
    ```
    *   **Transformer Modelleri:** Başlangıçta doğal dil işleme için geliştirilen "dikkat mekanizması" (attention mechanism) tabanlı bu modeller, zaman serisi tahmininde de son derece başarılı sonuçlar vermektedir. Özellikle çok uzun serilerdeki bağımlılıkları yakalamada LSTM'den daha etkili olabilirler.

---

### Referanslar

-   Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
-   Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.
-   Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735-1780.

---



**Pratik kısım için:**

1.  Bir CSV dosyasından veri okuyun ve `ts` nesnesine çevirin.
2.  Verinin başlangıç, bitiş ve frekansını bulun.
3.  Belirli bir dönemi çıkarın (`window` fonksiyonu kullanarak).
4.  Lag değerleri hesaplayın.
5.  Temel görselleştirme yapın (R `plot()` veya `ggplot2` ile).
6.  Tarih formatı dönüştürme işlemleri yapın.

**Örnek soru senaryosu:**
"Size bir CSV verilecek, içinde tarih ve değer sütunları olacak. Bu veriyi R'da okuyup uygun bir `ts` nesnesine çevirmeniz, ardından 2015-2020 yılları arasındaki veriyi çıkarmanız ve son olarak bu alt serinin grafiğini çizmeniz gerekecek."

**Pratik yapın!** Bu kodları çalıştırın, değiştirin, kırın ve tamir edin. Ancak bu şekilde öğrenebilirsiniz.


---