# R Programlama ve Zaman Serisi Analizine Giriş
## Ders Notları

**Dersin Amacı:** Bu derste R programlama dilinin temellerini öğrenecek ve zaman serisi analizine hazırlık yapacaksınız. Lütfen örnekleri kendi bilgisayarınızda uygulayın - okumak yeterli değildir.

---

## R'a Giriş ve Temel Yapılar

### Neden R Öğreniyoruz?

Bazılarınız "Python'u biliyorum, R'ı da öğrenmem gerekir mi?" diye soruyor. Açıkçası, zaman serisi analizi için R'ın avantajları var. ARIMA, Holt-Winters gibi klasik modeller için R'daki implementasyonlar Python'dakilerden daha olgun. Ama tabii, her iki dili de bilmek ideal.

R açık kaynak kodlu. Yani ücretsiz. Peki bu ne demek? Herkes kod yazabiliyor, paket geliştirebiliyor. Şu an 13,500'den fazla paket var. Karşılaştırma yapalım: 10 yıl önce 3,000 civarıydı. Üstel bir büyüme var.

**Önemli:** R'ı ve RStudio'yu mutlaka kurun. CRAN sitesinden R'ı, sonra RStudio'yu indirin. RStudio bir IDE - yani gelişmiş bir editör. R konsolu da var ama RStudio kullanmak hayatınızı kolaylaştırır.

### İlk Adımlar

R'ı açtığınızda konsol görürsünüz. Buraya direkt kod yazabilirsiniz:

```r
# Yorum satırı - R bunu çalıştırmaz
print("Merhaba")
```

Hash işareti yorum için kullanılır. Kodunuzu açıklayın, kendinize not alın. 6 ay sonra kendi kodunuzu anlamayabilirsiniz yoksa.

### Değişken Atama - Kritik Bir Nokta

R'da değişkenlere değer atamak için iki yöntem var:

```r
x <- 5
y = 5
```

İkisi de çalışır ama `<-` kullanın. Neden? R topluluğunda bu standart. Fonksiyon içinde parametre atarken `=` kullanılır, karışıklık olmasın diye.

Bir örnek:

```r
ogrenci_sayisi <- 45
ortalama_yas <- 21.5
ders_adi <- "Zaman Serisi Analizi"

# Değerleri görelim
print(ogrenci_sayisi)
```

**Dikkat:** R case-sensitive. Yani `Yas` ile `yas` farklı şeyler. Kod yazarken tutarlı olun.

### Operatörler

Matematiksel işlemler basit:

```r
# Temel aritmetik
10 + 5    # 15
10 - 5    # 5
10 * 5    # 50
10 / 5    # 2
10 ^ 2    # 100
10 %% 3   # 1 (mod, kalan)
```

Karşılaştırmalar:

```r
5 > 3     # TRUE
5 == 5    # TRUE (eşitlik için çift eşittir!)
5 != 3    # TRUE (eşit değil)
```

Mantıksal operatörler (bunlar çok kullanacaksınız):

```r
TRUE & FALSE   # FALSE (VE)
TRUE | FALSE   # TRUE (VEYA)
!TRUE          # FALSE (DEĞİL)
```

Şimdi gerçekçi bir örnek:

```r
sinav_notu <- 65
devamsizlik <- 3

# Geçti mi kontrolü
gecti <- (sinav_notu >= 60) & (devamsizlik < 5)
print(gecti)  # TRUE
```

### Pratik Uygulama

Kendiniz deneyin:

```r
# İki ürünün fiyatı
urun1 <- 150
urun2 <- 200

# İndirimli fiyatlar (%20 indirim)
indirimli1 <- urun1 * 0.8
indirimli2 <- urun2 * 0.8

# Toplam tutar
toplam <- indirimli1 + indirimli2
print(paste("Toplam:", toplam, "TL"))
```

---

## R'da Veri Yapıları

### Vektörler - R'ın Temeli

R'da her şey vektör. Tek bir sayı bile aslında 1 elemanlı vektör. Şunu deneyin:

```r
x <- 5
length(x)  # 1
```

Vektör oluşturmak için `c()` fonksiyonu kullanılır (combine kelimesinden gelir):

```r
notlar <- c(70, 85, 92, 68, 75)
isimler <- c("Ali", "Ayşe", "Mehmet")
mantiksal <- c(TRUE, FALSE, TRUE, TRUE)
```

Vektör işlemleri vektörize çalışır (bu çok önemli):

```r
notlar <- c(70, 85, 92, 68, 75)
notlar + 5        # Her nota 5 ekler
notlar * 1.1      # %10 artırır
notlar > 75       # Her eleman için kontrol
```

Bu Python'daki listelerden farklı. R'da vektör işlemleri çok hızlı çünkü altta C kodu çalışıyor.

**Vektör İndeksleme:**

```r
notlar <- c(70, 85, 92, 68, 75)

notlar[1]        # İlk eleman (dikkat: R'da indeks 1'den başlar!)
notlar[c(1,3)]   # 1. ve 3. elemanlar
notlar[-1]       # 1. eleman hariç hepsi
notlar[notlar > 75]  # 75'ten büyük olanlar
```

Python'dan geliyorsanız dikkat: R'da indeks 0 değil 1'den başlar.

### Data Frame - Gerçek Veri

Data frame, Excel'deki tablolar gibi. Farklı tipte sütunlar içerebilir:

```r
ogrenciler <- data.frame(
  ad = c("Ali", "Ayşe", "Mehmet", "Fatma"),
  yas = c(20, 22, 21, 23),
  not = c(85, 92, 78, 88),
  gecti = c(TRUE, TRUE, TRUE, TRUE)
)

print(ogrenciler)
```

**Çıktı:**
```
     ad yas not gecti
1    Ali  20  85  TRUE
2   Ayşe  22  92  TRUE
3 Mehmet  21  78  TRUE
4  Fatma  23  88  TRUE
```

Veriyi incelemek için kullanışlı fonksiyonlar:

```r
str(ogrenciler)      # Yapı bilgisi
head(ogrenciler, 2)  # İlk 2 satır
summary(ogrenciler)  # Özet istatistikler
dim(ogrenciler)      # Boyutlar (satır, sütun)
names(ogrenciler)    # Sütun isimleri
```

**Veri Çekme - Önemli:**

```r
# Sütun seçimi - 3 yöntem (hepsi aynı sonucu verir)
ogrenciler$not              # En yaygın
ogrenciler[, "not"]         # Sütun ismiyle
ogrenciler[, 3]             # İndeksle

# Satır seçimi
ogrenciler[1, ]             # İlk satır
ogrenciler[c(1,3), ]        # 1. ve 3. satırlar

# Hem satır hem sütun
ogrenciler[1:2, c("ad", "not")]

# Filtreleme (bunu çok kullanacaksınız)
yuksek_notlar <- ogrenciler[ogrenciler$not > 85, ]
print(yuksek_notlar)
```

### Gerçekçi Örnek

```r
# Aylık satış verisi
satis <- data.frame(
  ay = c("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran"),
  miktar = c(1200, 1350, 1180, 1420, 1580, 1650),
  hedef = c(1300, 1300, 1300, 1400, 1500, 1600)
)

# Hedefi aşan aylar
basarili <- satis[satis$miktar > satis$hedef, ]
print(basarili)

# Ortalama satış
mean(satis$miktar)

# Sapma oranı (hedeften)
satis$sapma_yuzde <- ((satis$miktar - satis$hedef) / satis$hedef) * 100
print(satis)
```

---

## 3. Hafta: Paketler ve Veri İçe Aktarma

### R Paketleri

R'ın gücü paketlerinden gelir. Temel R (base R) bile güçlü ama gerçek iş paketlerle yapılır.

**Paket kurulumu (sadece bir kez):**

```r
install.packages("forecast")
install.packages("dplyr")
```

**Paketi yükleme (her oturum için):**

```r
library(forecast)
library(dplyr)
```

Bunları karıştırmayın. Install bir kez, library her oturumda.

**Hangi paketlere ihtiyacınız var?**

Zaman serisi analizi için temel paketler:

```r
# Veri manipülasyonu
install.packages("dplyr")
install.packages("lubridate")  # Tarih/zaman

# Zaman serisi
install.packages("forecast")
install.packages("TSstudio")

# Görselleştirme
install.packages("ggplot2")
install.packages("plotly")
```

Paket versiyonunu kontrol edin:

```r
packageVersion("forecast")  # Versiyon önemli, bazı fonksiyonlar eski versiyonlarda yok
```

### Veri İçe Aktarma

**CSV dosyası:**

```r
# Yerel dosya
veriler <- read.csv("veri.csv", stringsAsFactors = FALSE)

# İnternetten
url <- "https://raw.githubusercontent.com/dosya/veri.csv"
veriler <- read.csv(url, stringsAsFactors = FALSE)
```

`stringsAsFactors = FALSE` önemli. Yoksa metinleri otomatik factor yapar, genelde istemezsiniz.

**Excel dosyası:**

```r
library(readxl)
veriler <- read_excel("veri.xlsx")
```

**Paket dataseti:**

```r
library(TSstudio)
data(USgas)  # Paket içinden yükler
```

Gerçek örnek:

```r
# İnternetten satış verisi çekelim
url <- "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Time-Series-Analysis-with-R/master/Chapter%201/TOTALNSA.csv"
satis_verisi <- read.csv(url, stringsAsFactors = FALSE)

# Yapıya bakalım
str(satis_verisi)
head(satis_verisi, 10)

# Sütun isimlerini düzeltelim
names(satis_verisi) <- c("tarih", "satis")
```

### Temel İstatistikler

Bir veri yüklediniz, şimdi ne yapacaksınız?

```r
# Özet istatistikler
summary(veriler)
mean(veriler$satis)
median(veriler$satis)
sd(veriler$satis)        # Standart sapma
var(veriler$satis)       # Varyans
quantile(veriler$satis)  # Yüzdelikler
```

Eksik veri varsa:

```r
mean(veriler$satis, na.rm = TRUE)  # NA'ları çıkar
```

### Basit Görselleştirme

```r
# Histogram
hist(veriler$satis, 
     main = "Satış Dağılımı",
     xlab = "Satış Miktarı",
     col = "lightblue")

# Kutu grafiği
boxplot(veriler$satis,
        main = "Satış Kutu Grafiği")

# Zaman serisi grafiği
plot(veriler$satis, type = "l",
     main = "Zaman İçinde Satış",
     ylab = "Satış",
     xlab = "Gözlem")
```

---

## Ödev ve Pratik

**Bu haftanın ödevi (zorunlu):**

1. Kendi bir data frame oluşturun (en az 20 satır, 4 sütun)
2. Bu verinin özetini çıkarın
3. En az 3 filtreleme yapın
4. Ortalama, medyan, standart sapma hesaplayın
5. Bir histogram çizdirin

**Örnek veri fikirleri:**
- Günlük hava durumu
- Haftalık market harcamaları
- Aylık elektrik faturaları
- Kurs notları

**Dikkat edilecek noktalar:**

- Kodunuza yorum ekleyin
- Değişken isimlerini anlamlı seçin
- Her adımı kontrol edin (`head()`, `str()` kullanın)
- Hata alırsanız mesajı okuyun, Google'da aratın

**Sık yapılan hatalar:**

1. `<-` yerine `=` kullanmak (kabul edilir ama standart değil)
2. İndeksleri karıştırmak (R'da 1'den başlar!)
3. Parantez kapatmayı unutmak
4. Fonksiyon ismini yanlış yazmak (R case-sensitive)

---

## Gelecek Hafta

Gelecek hafta tarih ve zaman nesnelerini işleyeceğiz. Bu zaman serisi için kritik. Hazırlık:

```r
# Bu fonksiyonlara bakın
Sys.Date()
Sys.time()
as.Date("2024-01-01")
```

**Kaynaklar:**

- R Documentation: `?fonksiyon_ismi`
- Stack Overflow: En çok kullanacağınız kaynak
- CRAN: Paket dokümantasyonları

**Son not:** R öğrenmek pratikle olur. Her gün en az 30 dakika kod yazın. Okumak yetmez.

Sorularınız olursa ofis saatlerinde gelin veya email atın.