[İçindekiler](Course_notes.md) · [← Ek A: R'a Giriş](R_Giris.md)

## Ek B. Gretl'e Giriş

Bu ek, derste kullanılan ekonometri yazılımı Gretl'i menüler üzerinden tanıtır ve iki uygulama içerir: `AirPassengers` serisi için SARIMA modeli ve öngörü (B.2–B.4), günlük borsa getirilerinde artıklara dayalı anomali tespiti (B.5). Gretl'in komut dili ve VAR uygulaması Bölüm 11'de ayrıntılı olarak ele alınır. Ekteki bütün veri dosyaları Gretl kurulumuyla birlikte gelir; ayrıca bir şey indirmek gerekmez.

---

### B.1. Gretl Nedir?

Gretl (*Gnu Regression, Econometrics and Time-series Library*), ekonometri ve zaman serisi analizi için geliştirilmiş, açık kaynak kodlu ve ücretsiz bir yazılımdır. C dilinde yazılmıştır ve GNU Genel Kamu Lisansı (GPL) ile dağıtılır; R ya da Python üzerine kurulu değildir, kendi hesaplama kütüphanesini kullanır. Başlıca geliştiricileri Allin Cottrell ve Riccardo "Jack" Lucchetti'dir.

Gretl'in iki kullanım biçimi vardır:

- **Grafik arayüz** (GUI, *graphical user interface*): Veri açma, grafik çizme, model kurma ve öngörü gibi işlemler menülerden yapılır. Kod yazmadan analiz yapmak isteyenler için uygundur.
- **Komut dili hansl**: Adı, "Hansl's a Neat Scripting Language" ifadesinin özyinelemeli (kendine gönderme yapan) kısaltmasıdır. Menülerden yapılan her işlemin bir komut karşılığı vardır; betikler (`.inp` dosyaları) analizlerin tekrarlanabilir olmasını sağlar (Bölüm 11).

Gretl'in eğitimdeki bir avantajı, çıktılarının ders kitaplarındaki gösterime yakın olmasıdır: katsayı tablosu, standart hatalar, p-değerleri ve bilgi kriterleri tek pencerede, açık adlarıyla raporlanır.

---

### B.2. Veriyi Açmak ve Görselleştirmek

Uygulamada Box ve Jenkins'in "Series G" verisi, yani ders boyunca kullanılan `AirPassengers` serisi kullanılır. Gretl bu seriyi `bjg` adıyla örnek veri dosyaları arasında getirir (Windows'ta `C:\Program Files\gretl\data\misc\bjg.gdt`).

**Veriyi açmak:**

1. Gretl başlatılır ve **File → Open data → Sample file…** seçilir.
2. Açılan pencerenin **Gretl** sekmesinde, açıklaması "Box and Jenkins Series G (airline passengers)" olan `bjg` satırına çift tıklanır. Dosya kurulumla geldiği için **Look on server** ya da **Install** düğmelerine gerek yoktur.

Aynı işin komut karşılığı `open bjg` satırıdır (**Tools → Gretl console** ya da bir betik penceresinde).

Ana pencerede iki değişken görünür: `g` (aylık uluslararası havayolu yolcu sayısı, bin kişi) ve `lg` (`g`'nin doğal logaritması). Pencerenin alt kısmında verinin aylık (*Monthly*) olduğu ve 1949:01–1960:12 arasında 144 gözlem içerdiği okunur.

**Zaman grafiği:** `g` seçilip sağ tıklanır ve **Time series plot** seçilir (ya da **View → Graph specified vars → Time series plot…**). Grafikte Bölüm 2'de tanımlanan üç özellik görülür:

- **Trend:** Yolcu sayısı yıllar içinde sürekli artar; serinin ortalaması sabit değildir.
- **Mevsimsellik:** Her yıl yaz aylarında tepe, kış aylarında dip vardır.
- **Artan varyans:** Mevsimsel dalgaların boyu seviyeyle birlikte büyür. Bu, serinin toplamsal değil çarpımsal yapıda olduğunu gösterir (Bölüm 2.4).

---

### B.3. Log Dönüşümü ve Model Kurma

**Log dönüşümü:** Çarpımsal yapıyı toplamsala çevirmek ve artan varyansı dengelemek için serinin doğal logaritması alınır (Bölüm 2.4). `g` seçiliyken **Add → Logs of selected variables** seçilir; ana pencereye `l_g` adlı yeni bir değişken eklenir. Değerleri dosyada hazır gelen `lg` ile aynıdır. Yine de `l_g`'nin bu yolla oluşturulması yararlıdır: Gretl bu değişkenin `g`'nin logaritması olduğunu bilir ve öngörü aşamasında sonuçları doğrudan yolcu sayısı olarak verebilir (B.4). `l_g`'nin grafiğinde mevsimsel dalgaların boyu yaklaşık sabittir; trend ve mevsimsellik ise sürmektedir. Bunlar modeldeki fark alma işlemleriyle giderilir.

**Model:** `AirPassengers` için Box ve Jenkins'in önerdiği havayolu modeli, yani ARIMA(0,1,1)(0,1,1)[12] kullanılır (Bölüm 7.3). Modelin parçaları şunlardır:

- Log ölçek: artan varyansı dengelemek için.
- $d = 1$: trendi gidermek için birinci fark, $x_t - x_{t-1}$.
- $D = 1$: yıllık deseni gidermek için mevsimsel fark, $x_t - x_{t-12}$.
- $q = 1$ ve $Q = 1$: geçen ayın ve geçen yılın aynı ayının sürprizleriyle düzeltme yapan birer MA terimi.

Modelin seçimi normalde farkı alınmış serinin ACF ve PACF grafiklerinden yapılır (Bölüm 6.6 ve 7.6.4); bu seri için korelogramlar da aynı adayı gösterir.

**Menüden model kurma:**

1. **Model → Univariate time series → ARIMA…** seçilir (Gretl'in eski sürümlerinde alt menünün adı "Time series" idi).
2. **Dependent variable** olarak `l_g` seçilir; orijinal `g` değil, log serisi kullanılır.
3. Mevsimsel olmayan kısım: AR 0, Difference 1, MA 1. Mevsimsel kısım: AR 0, Difference 1, MA 1. Mevsim uzunluğu 12'dir; Gretl bunu verinin frekansından kendisi önerir.
4. **Include a constant** kutusunun işareti kaldırılır. Bir normal ve bir mevsimsel farkı alınmış seride sabit terim, giderek hızlanan bir trend anlamına gelir ve gereksizdir. Model kutu işaretliyken kurulduğunda `const` katsayısının p-değeri 0.8675'tir (anlamsız) ve AIC −483.39'dan −481.42'ye kötüleşir.
5. **OK** ile model kestirilir.

Komut karşılığı:

```gretl
open bjg
logs g                            # l_g değişkenini oluşturur
arima 0 1 1 ; 0 1 1 ; l_g --nc    # (p d q) ; (P D Q) ; bağımlı değişken; --nc: sabit yok
```

**Çıktının okunması:** Pencerenin başlığı "Model 1: ARIMA, using observations 1950:02-1960:12 (T = 131)" biçimindedir. Gözlemler 1950:02'den başlar, çünkü bir normal ve bir mevsimsel fark ilk 13 ayı tüketir.

| Satır | Katsayı | Standart hata | p-değeri |
| --- | --- | --- | --- |
| `theta_1` ($\theta_1$) | −0.4018 | 0.0896 | 7.38e-06 |
| `Theta_1` ($\Theta_1$) | −0.5569 | 0.0731 | 2.57e-14 |

İki MA katsayısı da %1 düzeyinde anlamlıdır (`***`). Pencerenin alt kısmında log olabilirlik 244.6965, Akaike ölçütü (AIC) −483.3930 ve Schwarz ölçütü (BIC) −474.7674'tür. Bu değerler R'ın `auto.arima()` sonucuyla aynıdır (Bölüm 7.6.5). Bilgi kriterleri yalnızca aynı veri ve aynı fark sayılarıyla kurulmuş modelleri karşılaştırmak için kullanılır; küçük değer daha iyi uyumu gösterir (Bölüm 7.4).

---

### B.4. Öngörü ve Yelpaze Grafiği

Gretl öngörüyü yalnızca veri setinde tanımlı satırlar için üretir. Veri Aralık 1960'ta bittiği için 1961'i tahmin etmeden önce veri setine 12 boş satır eklenir:

1. **Data → Add observations…** seçilir ve 12 yazılır. Gretl yeni satırları 1961:01–1961:12 olarak tarihlendirir; bu satırlarda `g` ve `l_g` boştur.
2. Model penceresinde **Analysis → Forecasts…** seçilir.
3. Açılan pencerede öngörü aralığı (1961:01–1961:12) kendiliğinden seçilidir. **Produce forecast for** seçeneğinde `g` işaretlenirse Gretl log ölçekteki öngörüleri `exp()` ile geri çevirir ve sonuçları doğrudan yolcu sayısı olarak verir. **Plot confidence interval using** seçeneğinde **shaded area** seçilirse aralıklar gölgeli bir bant olarak çizilir.
4. **OK** ile öngörü tablosu ve grafik açılır. Öngörüler **Save forecast** ile bir değişken olarak saklanabilir.

Komut karşılığı:

```gretl
dataset addobs 12
fcast 1961:01 1961:12 --plot=display
```

Log ölçekte Ocak 1961 öngörüsü 6.11019, %95 aralığı 6.03822–6.18215'tir. Geri çevrildiğinde nokta öngörüsü $e^{6.11019} \approx 450.4$, aralık yaklaşık 419.1–484.0 bin yolcudur; Bölüm 7.6.7'deki R sonucuyla aynıdır. Temmuz 1961 öngörüsü $e^{6.50729} \approx 670.0$ bin yolcudur. Log ölçekteki öngörüler 6.05 ile 6.51 arasında değişir.

Grafikte ortadaki çizgi nokta öngörüleridir ve geçmiş yıllardaki mevsimsel deseni sürdürür. Gölgeli bant ufuk uzadıkça genişler: Aralık 1961 için %95 aralığı yaklaşık 406.7–560.0 bin yolcudur. Bu görünüme literatürde **yelpaze grafiği** (*fan chart*) denir. Bant, "gerçek değerin %95 olasılıkla bu sınırlar arasında kalması beklenir" biçiminde okunur.

**Not —** Log ölçekte kurulan bir modelin öngörüsü `exp()` ile geri çevrildiğinde, orijinal ölçekte ortalamayı değil ortancayı verir (Bölüm 7.6.7). Model, Gretl'in `logs` komutuyla değil de dosyada hazır gelen `lg` ile kurulmuşsa **Produce forecast for** seçeneği görünmez; bu durumda öngörü kaydedilir ve **Add → Define new variable** ile `yolcu_tahmin = exp(forecast_adi)` biçiminde orijinal ölçeğe çevrilir.

---

### B.5. Artıklara Dayalı Anomali Tespiti

**Anomali** (*anomaly*) ya da aykırı değer (*outlier*), beklenen davranıştan belirgin biçimde sapan gözlemdir (Bölüm 1.3). Bu bölümde basit bir yöntem uygulanır: Serinin olağan davranışı bir modelle açıklanır, modelin açıklayamadığı kısım, yani **artık** (*residual*) incelenir:

$$
\text{artık} = \text{gözlenen değer} - \text{modelin uydurduğu değer}
$$

Olağan dönemlerde artıklar sıfır çevresinde küçük değerler alır; modelin açıklayamadığı ani olaylarda ise büyür. Veri olarak Gretl'in örnek dosyalarındaki `djclose` kullanılır: Dow Jones endeksinin 1980–1989 arasındaki günlük kapanış değerleri (2528 gözlem; Bölüm 3.2'deki "Hemen deneyin 3.2e" kutusunda da kullanılır).

**1. Getiri serisi:** Fiyat serisi durağan değildir; bu yüzden günlük log getiri kullanılır. **Add → Define new variable** ile `getiri = 100 * diff(log(djclose))` tanımlanır (yüzde olarak günlük log getiri; Bölüm 3, "Kendinizi test edin" 3. soru).

**2. Olağan davranışın modeli:** Getirinin bugünkü değeri bir AR(1) modeliyle açıklanır: $\text{getiri}_t = \alpha + \beta\,\text{getiri}_{t-1} + u_t$. Menüden **Model → Ordinary Least Squares** seçilir, bağımlı değişken `getiri`, açıklayıcı değişkenler `const` ve `getiri`'nin birinci gecikmesi olarak verilir (gecikme, **Add → Lags of selected variables** ile `getiri_1` adıyla eklenebilir).

**3. Artıkları kaydetmek:** Model penceresinde **Save → Residuals** seçilir; artıklar yeni bir değişken olarak veri setine eklenir.

**4. Eşik:** Bir değişken normal dağılıyorsa değerlerinin yaklaşık %99.7'si ortalamanın ±3 standart sapma **içinde** kalır (*üç sigma kuralı*). Mutlak değeri 3 standart sapmayı aşan artıklar olası anomali olarak işaretlenir. **Add → Define new variable** ile `bayrak = abs(u) > 3 * sd(u)` tanımlanır; bu değişken anomali olan günlerde 1, diğer günlerde 0 değerini alır.

Bütün adımların komut karşılığı:

```gretl
open djclose
series getiri = 100 * diff(log(djclose))   # günlük log getiri (%)
ols getiri const getiri(-1)                # AR(1): getiri(-1) bir gün önceki değerdir
series u = $uhat                           # artıklar
scalar s = sd(u)                           # artıkların standart sapması
series bayrak = abs(u) > 3 * s             # 3 sigma kuralı
scalar n = sum(bayrak)                     # işaretlenen gün sayısı
print s n
smpl bayrak == 1 --restrict                # yalnızca işaretlenen günler
print getiri u --byobs
```

**Sonuçlar:** Model 2526 gözlemle kestirilir; `getiri_1` katsayısı 0.0197'dir ve p-değeri 0.3216 olduğu için anlamlı değildir: Günlük getiri, bir önceki günün getirisinden neredeyse hiç etkilenmez. Artıkların standart sapması 1.1556, eşik $3 \times 1.1556 \approx 3.467$'dir. 2526 günün 24'ü (%0.95) işaretlenir. En büyük artık 19 Ekim 1987'dedir ("Kara Pazartesi"): Getiri −25.63, artık −25.59'dur. Ekim 1987'nin başka günleri de listededir (örneğin 20 Ekim'de +5.72, 21 Ekim'de +9.67): Büyük şoklar zaman içinde kümelenir.

**Yorum:** Artıklar normal dağılsaydı 2526 günde ortalama yaklaşık $2526 \times 0.0027 \approx 6.8$ gün eşiği aşardı; işaretlenen 24 gün bunun üç katından fazladır. Bu, finansal getirilerin normal dağılıma göre daha kalın kuyruklu olduğunu, yani büyük hareketlerin normal dağılımın öngördüğünden sık görüldüğünü gösterir. Üç sigma kuralı bu yüzden kesin bir anomali tanımı değil, incelenecek günleri seçen kaba bir süzgeçtir. İşaretlenen günlerin gerçekten olağan dışı olup olmadığına, o günlere ait bilgiyle (haberler, piyasa olayları) karar verilir.

---

### B.6. Özet Tablo

| İşlem | Menü yolu | Komut | Kavram |
| --- | --- | --- | --- |
| Örnek veri açma | File → Open data → Sample file | `open bjg` | Veri okuma |
| Zaman grafiği | View → Graph specified vars → Time series plot | `gnuplot g --time-series --with-lines` | Görsel inceleme |
| Log dönüşümü | Add → Logs of selected variables | `logs g` | Varyansı dengeleme (Bölüm 2.4) |
| ARIMA kurma | Model → Univariate time series → ARIMA | `arima 0 1 1 ; 0 1 1 ; l_g --nc` | Box-Jenkins kestirimi (Bölüm 7.4) |
| Gözlem ekleme | Data → Add observations | `dataset addobs 12` | Öngörü için boş dönem |
| Öngörü | Analysis → Forecasts | `fcast` | Örneklem dışı öngörü |
| Artık kaydetme | Save → Residuals | `series u = $uhat` | Artık analizi |
| Yeni değişken | Add → Define new variable | `series x = ...` | Dönüşüm, kukla değişken |

Son olarak model seçiminde **tutumluluk ilkesi** (*principle of parsimony*) hatırlanmalıdır: Veriyi yeterince iyi açıklayan modeller arasında en az parametreli olan tercih edilir (Bölüm 7.4'teki AIC karşılaştırması).

### B.7. Kaynaklar

- Cottrell, A. ve Lucchetti, R. *Gretl User's Guide*. Gretl ile birlikte gelir (**Help → User's guide**).
- Cottrell, A. ve Lucchetti, R. *Gretl Command Reference*. Gretl ile birlikte gelir (**Help → Command reference**).
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. ve Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5. baskı). Wiley.
- Hyndman, R. J. ve Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3. baskı). OTexts.

---

[İçindekiler](Course_notes.md) · [← Ek A: R'a Giriş](R_Giris.md)
