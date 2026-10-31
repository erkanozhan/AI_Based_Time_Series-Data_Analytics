[İçindekiler](Course_notes.md) · [Ek B: Gretl'e Giriş →](Gretl.md)

## Ek A. R'a Giriş

Bu ek, ders notlarında kullanılan R (*R programming language*) komutlarını ilk kez görenler için kısa bir başlangıç kılavuzudur. Değişken atama, temel işlemler, vektörler, veri çerçeveleri, paketler, veri okuma ve temel grafikler sırasıyla ele alınır. Tarih ve zaman nesneleri Bölüm 4'te, zaman serisi nesneleri (`ts`, `xts`) Bölüm 5'te ayrıntılı olarak işlenir. Ekteki bütün kodlar depo kök dizininden çalıştırılacak biçimde yazılmıştır; gerekli paketlerin kurulumu [`Codes/README.md`](Codes/README.md) dosyasında anlatılır.

---

### A.1. Neden R?

R, istatistiksel hesaplama ve grafik için geliştirilmiş, açık kaynak kodlu ve ücretsiz bir programlama dilidir. Zaman serisi analizinde R'ın başlıca avantajı, ARIMA, üstel düzeltme (Holt-Winters) ve ayrıştırma gibi klasik istatistiksel yöntemler için uzun süredir geliştirilen, olgun paketlere sahip olmasıdır. Python ise makine öğrenmesi ve derin öğrenme kütüphaneleriyle öne çıkar; bu nedenle derste iki dil birlikte kullanılır.

Paketler, R'ın resmî paket arşivi CRAN (*Comprehensive R Archive Network*) üzerinden dağıtılır. CRAN'da bugün 20 000'i aşkın paket bulunmaktadır.

R'ın öne çıkan özellikleri şunlardır:

- Açık kaynak kodludur ve ücretsizdir.
- Geniş bir paket ekosistemine sahiptir.
- Güçlü görselleştirme araçları sunar.
- Etkin bir kullanıcı topluluğu vardır.
- Zaman serisi analizi için özel nesne ve fonksiyonlar içerir.

R'ın yanında genellikle RStudio kullanılır. RStudio bir tümleşik geliştirme ortamıdır (IDE, *integrated development environment*): Kod düzenleyici, konsol, grafik paneli ve çalışma alanı aynı pencerede toplanır. Önce CRAN'dan R, ardından RStudio kurulur.

### A.2. İlk Adımlar

R açıldığında komutların yazıldığı bir konsol görünür. Komutlar doğrudan konsola yazılabilir:

```r
# Yorum satırı: R bu satırı çalıştırmaz
print("Merhaba")
#> [1] "Merhaba"
```

`#` işaretinden sonra gelen metin **yorumdur** (*comment*). Yorumlar kodun ne yaptığını açıklamak için yazılır; birkaç ay sonra kodu yeniden okuyan kişi için büyük kolaylık sağlar. Ders notlarında `#>` ile başlayan satırlar ise R'ın ekrana yazdığı çıktıyı gösterir; konsola yazılmaz.

### A.3. Değişken Atama

R'da bir değere ad vermek için iki işaret kullanılabilir:

```r
x <- 5
y = 5
```

İkisi de çalışır; ancak R topluluğunda atama için `<-` kullanmak yerleşik bir alışkanlıktır. `=` işareti çoğunlukla fonksiyon çağrısında argüman verirken kullanılır (`round(3.14159, digits = 2)` gibi); iki kullanımın ayrı tutulması kodun okunurluğunu artırır.

```r
ogrenci_sayisi <- 45
ortalama_yas   <- 21.5
ders_adi       <- "Zaman Serisi Analizi"
print(ogrenci_sayisi)
#> [1] 45
```

**Dikkat:** R büyük-küçük harfe duyarlıdır (*case-sensitive*): `Yas` ile `yas` iki ayrı addır. Aynı kural fonksiyon ve veri seti adları için de geçerlidir.

### A.4. Operatörler

Aritmetik işlemler:

```r
x <- 10
y <- 2
x + y    # toplama
#> [1] 12
x - y    # çıkarma
#> [1] 8
x * y    # çarpma
#> [1] 20
x / y    # bölme
#> [1] 5
x ^ 2    # üs alma
#> [1] 100
x %% 3   # bölümden kalan (mod)
#> [1] 1
```

Karşılaştırma operatörleri sonuç olarak mantıksal bir değer (`TRUE` ya da `FALSE`) döndürür:

```r
a <- 5
b <- 7
a == b   # eşit mi? (eşitlik için çift eşittir)
#> [1] FALSE
a != b   # eşit değil mi?
#> [1] TRUE
a > b    # büyük mü?
#> [1] FALSE
a <= 5   # küçük ya da eşit mi?
#> [1] TRUE
```

Mantıksal operatörler koşulları birleştirir: `&` "ve", `|` "veya", `!` "değil" anlamına gelir.

```r
sinav_notu  <- 65
devamsizlik <- 3
gecti <- (sinav_notu >= 60) & (devamsizlik < 5)   # iki koşul da sağlanıyor mu?
print(gecti)
#> [1] TRUE
```

Kısa bir uygulama olarak iki ürünün %20 indirimli toplam fiyatı şöyle hesaplanır:

```r
urun1 <- 150
urun2 <- 200
toplam <- urun1 * 0.8 + urun2 * 0.8
print(paste("Toplam:", toplam, "TL"))
#> [1] "Toplam: 280 TL"
```

---

### A.5. Veri Yapıları

#### A.5.1. Vektörler

R'da temel veri yapısı **vektördür** (*vector*): aynı türden değerlerin sıralı bir listesi. Tek bir sayı bile bir elemanlı bir vektördür:

```r
x <- 5
length(x)
#> [1] 1
```

Vektör oluşturmak için `c()` fonksiyonu kullanılır (*combine*, birleştir):

```r
notlar    <- c(70, 85, 92, 68, 75)
isimler   <- c("Ali", "Ayşe", "Mehmet")
mantiksal <- c(TRUE, FALSE, TRUE, TRUE)
```

Vektör işlemleri eleman eleman yapılır (*vektörel işlem*); döngü yazmaya gerek kalmaz:

```r
notlar + 5      # her nota 5 ekler
#> [1] 75 90 97 73 80
notlar * 1.1    # her notu %10 artırır
#> [1]  77.0  93.5 101.2  74.8  82.5
notlar > 75     # her eleman için koşul
#> [1] FALSE  TRUE  TRUE FALSE FALSE
```

Köşeli parantez vektörden eleman seçer. R'da sıra numaraları 1'den başlar (Python'da 0'dan):

```r
notlar[1]             # ilk eleman
#> [1] 70
notlar[c(1, 3)]       # 1. ve 3. elemanlar
#> [1] 70 92
notlar[-1]            # 1. eleman hariç hepsi
#> [1] 85 92 68 75
notlar[notlar > 75]   # 75'ten büyük olanlar
#> [1] 85 92
```

#### A.5.2. Veri Çerçeveleri

**Veri çerçevesi** (*data frame*), her sütunu bir değişken, her satırı bir gözlem olan tablodur; sütunlar farklı türlerde olabilir:

```r
ogrenciler <- data.frame(
  ad    = c("Ali", "Ayşe", "Mehmet", "Fatma"),
  yas   = c(20, 22, 21, 23),
  not   = c(85, 92, 78, 88),
  gecti = c(TRUE, TRUE, TRUE, TRUE)
)
print(ogrenciler)
#>       ad yas not gecti
#> 1    Ali  20  85  TRUE
#> 2   Ayşe  22  92  TRUE
#> 3 Mehmet  21  78  TRUE
#> 4  Fatma  23  88  TRUE
```

Bir veri çerçevesini incelemek için sık kullanılan fonksiyonlar: `str()` (yapı), `head()` (ilk satırlar), `summary()` (özet istatistikler), `dim()` (satır ve sütun sayısı) ve `names()` (sütun adları).

Sütun ve satır seçimi:

```r
ogrenciler$not                    # sütun adıyla (en yaygın yol)
#> [1] 85 92 78 88
ogrenciler[, "not"]               # [satır, sütun] biçiminde, sütun adıyla
ogrenciler[, 3]                   # aynı sütun, sıra numarasıyla
ogrenciler[1, ]                   # ilk satır
ogrenciler[1:2, c("ad", "not")]   # ilk iki satırın iki sütunu
ogrenciler[ogrenciler$not > 85, ] # koşula göre süzme
#>      ad yas not gecti
#> 2  Ayşe  22  92  TRUE
#> 4 Fatma  23  88  TRUE
```

Köşeli parantez içinde virgülün solu satırları, sağı sütunları seçer; boş bırakılan taraf "hepsi" anlamına gelir.

Aylık bir satış tablosunda hedefi aşan aylar ve hedeften sapma oranı şöyle bulunur:

```r
satis <- data.frame(
  ay     = c("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran"),
  miktar = c(1200, 1350, 1180, 1420, 1580, 1650),
  hedef  = c(1300, 1300, 1300, 1400, 1500, 1600)
)
satis[satis$miktar > satis$hedef, "ay"]    # hedefi aşan aylar
#> [1] "Şubat"   "Nisan"   "Mayıs"   "Haziran"
mean(satis$miktar)                         # ortalama satış
#> [1] 1396.667
satis$sapma_yuzde <- round((satis$miktar - satis$hedef) / satis$hedef * 100, 1)
satis$sapma_yuzde
#> [1] -7.7  3.8 -9.2  1.4  5.3  3.1
```

---

### A.6. Paketler

R'ın temel kurulumuyla gelen fonksiyonlar (*base R*) geniş bir yelpazeyi kapsar; özel yöntemler ise **paketlerle** eklenir. Bir paket bilgisayara bir kez kurulur, her yeni oturumda ise yüklenir:

```r
install.packages("forecast")   # kurulum: yalnızca bir kez
library(forecast)              # yükleme: her oturumda
packageVersion("forecast")     # kurulu sürümü gösterir
```

Bazı fonksiyonlar paketlerin yalnızca yeni sürümlerinde bulunduğu için sürüm bilgisi önemlidir. Derste kullanılan paketlerin tamamı tek komutla kurulabilir (`source("Codes/install_packages.R")`). Başlıcaları şunlardır:

| Paket | Kullanım amacı | İlk kullanıldığı bölüm |
| --- | --- | --- |
| `stats` | Temel istatistik fonksiyonları (`acf()`, `arima()`, `decompose()`); R ile birlikte gelir | 2 |
| `tseries` | Durağanlık testleri (`adf.test()`, `kpss.test()`) | 3 |
| `lubridate` | Tarih ve zaman işlemleri | 4 |
| `xts`, `zoo` | Düzensiz aralıklı zaman serileri | 5 |
| `TSstudio` | `USgas` veri seti ve `ts_info()` | 5 |
| `ggplot2` | Gelişmiş grafikler | 6 |
| `forecast` | `auto.arima()`, `forecast()`, `accuracy()` | 7 |

### A.7. Veri Okuma

#### A.7.1. CSV Dosyası

Virgülle ayrılmış değer dosyaları (CSV, *comma-separated values*) `read.csv()` ile okunur. Depodaki `data/AirPassengers.csv` dosyası iki sütun içerir: `Month` (yıl-ay) ve `Passengers` (bin yolcu).

```r
ap <- read.csv("data/AirPassengers.csv")
str(ap)
#> 'data.frame':	144 obs. of  2 variables:
#>  $ Month     : chr  "1949-01" "1949-02" "1949-03" "1949-04" ...
#>  $ Passengers: int  112 118 132 129 121 135 148 148 136 119 ...
head(ap, 3)
#>     Month Passengers
#> 1 1949-01        112
#> 2 1949-02        118
#> 3 1949-03        132
```

R 4.0'dan beri `read.csv()` metin sütunlarını karakter (`chr`) olarak okur. Daha eski sürümlerde metinler varsayılan olarak faktöre (`factor`) çevrildiği için eski kaynaklarda `stringsAsFactors = FALSE` argümanı sıkça görülür; güncel sürümlerde bu argüman gerekmez.

Dosya internetteki bir adresten de okunabilir (örneğin Colab'da):

```r
url <- "https://raw.githubusercontent.com/erkanozhan/AI_Based_Time_Series-Data_Analytics/main/data/AirPassengers.csv"
ap <- read.csv(url)
```

Excel dosyaları için `readxl` paketindeki `read_excel("dosya.xlsx")` fonksiyonu kullanılır.

#### A.7.2. Paketlerle Gelen Veri Setleri

R ve bazı paketler hazır veri setleriyle gelir; `data()` bunları çalışma alanına yükler:

```r
data("AirPassengers")   # R ile gelir (datasets paketi)
data("iris")            # R ile gelir; ad küçük harfle yazılır
library(TSstudio)
data(USgas)             # TSstudio paketinden
```

Veri seti adları da büyük-küçük harfe duyarlıdır: `data("Iris")` yazıldığında R veri setini bulamaz ve uyarı verir.

### A.8. Temel İstatistikler

Okunan verinin özet istatistikleri şöyle hesaplanır:

```r
yolcu <- ap$Passengers
summary(yolcu)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>   104.0   180.0   265.5   280.3   360.5   622.0 
sd(yolcu)      # standart sapma
#> [1] 119.9663
var(yolcu)     # varyans
#> [1] 14391.92
range(yolcu)   # en küçük ve en büyük değer
#> [1] 104 622
```

**Varyans** (*variance*) değerlerin ortalama etrafındaki yayılımını ölçer: Her değerin ortalamadan farkı alınır, farkların kareleri toplanır ve gözlem sayısının bir eksiğine ($n - 1$) bölünür. **Standart sapma** (*standard deviation*) varyansın kareköküdür ve verinin kendi biriminde ifade edildiği için daha kolay yorumlanır. R'ın `var()` ve `sd()` fonksiyonları $n - 1$'e böler; $n$'ye bölen tanımla farkı Bölüm 2.2'de açıklanır.

Eksik değer (`NA`) içeren bir vektörde `mean(x)` sonucu da `NA` olur; eksik değerleri dışarıda bırakmak için `na.rm = TRUE` argümanı verilir:

```r
mean(c(10, 20, NA))
#> [1] NA
mean(c(10, 20, NA), na.rm = TRUE)
#> [1] 15
```

### A.9. Temel Grafikler

Görselleştirme, veriyi anlamanın en hızlı yoludur. Temel R ile üç sık kullanılan grafik şöyle çizilir:

```r
# Histogram: değerlerin hangi aralıklarda yoğunlaştığını gösterir
hist(yolcu, main = "Aylık yolcu sayısının dağılımı",
     xlab = "Yolcu (bin)", ylab = "Frekans", col = "lightblue")

# Kutu grafiği: ortanca, çeyrekler ve olası aykırı değerler
boxplot(yolcu, main = "Aylık yolcu sayısı", ylab = "Yolcu (bin)", col = "lightgreen")

# Çizgi grafiği: değerlerin zaman sırasına göre seyri
plot(yolcu, type = "l", main = "Aylık yolcu sayısı, 1949-1960",
     xlab = "Gözlem sırası (ay)", ylab = "Yolcu (bin)", col = "darkred", lwd = 2)
```

**Histogram** (*histogram*) değerlerin aralıklara göre dağılımını gösterir. **Kutu grafiği** (*box plot*) verinin ortancasını (medyan), orta %50'sini kapsayan kutuyu ve kutunun dışında kalan olası aykırı değerleri özetler. **Çizgi grafiği** zaman serisi analizinin ilk adımıdır: Trend ve mevsimsellik çoğu zaman ilk bakışta fark edilir (Bölüm 1.5). Burada yatay eksen yalnızca gözlem sırasıdır; tarih bilgisini taşıyan `ts` nesnesiyle çizim Bölüm 5'te ele alınır.

### A.10. Alıştırmalar

1. En az 20 satır ve 4 sütundan oluşan bir veri çerçevesi oluşturun (örneğin haftalık market harcamaları, aylık elektrik faturaları ya da günlük hava sıcaklığı).
2. `str()`, `head()` ve `summary()` ile verinin özetini çıkarın.
3. En az üç farklı koşulla satır süzün.
4. Bir sütunun ortalamasını, ortancasını ve standart sapmasını hesaplayın.
5. Aynı sütunun histogramını ve çizgi grafiğini çizin.

**Sık yapılan hatalar:**

1. Sıra numarasını 0'dan başlatmak (R'da 1'den başlar).
2. Büyük-küçük harfi karıştırmak (`packageversion()` yerine `packageVersion()`, `data("Iris")` yerine `data("iris")`).
3. Açılan parantezi kapatmayı unutmak.
4. Paketi `library()` ile yüklemeden fonksiyonunu çağırmak.

Yardım için konsola `?fonksiyon_adı` (örneğin `?mean`) yazılır; paketlerin belgeleri CRAN sayfalarında bulunur.

Sonraki adım, tarih ve zaman nesneleridir: [Bölüm 4. R'da Tarih ve Zaman Nesneleri](notlar/04_rda_tarih_ve_zaman_nesneleri.md).

---

[İçindekiler](Course_notes.md) · [Ek B: Gretl'e Giriş →](Gretl.md)
