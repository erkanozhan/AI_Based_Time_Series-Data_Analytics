[İçindekiler](../Course_notes.md) · [← 6. Veri Manipülasyonu, Görselleştirme ve ACF/PACF](06_veri_manipulasyonu_gorsellestirme_ve_acf_pacf.md) · [8. Model Değerlendirme: Hata Metrikleri ve Eğitim-Test Ayrımı →](08_model_degerlendirme_hata_metrikleri_ve_egitim_test_ayrimi.md)

## 7. Klasik İstatistiksel Modeller: ARIMA, SARIMA ve Üstel Düzeltme

Önceki bölümlerde veriyi tanımayı, `ts` nesnesine dönüştürmeyi, görselleştirmeyi ve ACF/PACF grafiklerini okumayı öğrendik. Artık modelleme aşamasına geçebiliriz.

Bu bölümde zaman serisi analizinin "klasik" temelini oluşturan **ARIMA** ailesini ve bölümün sonunda (7.8) onun güçlü bir alternatifi olan **üstel düzeltme (ETS)** ailesini inceleyeceğiz. Bu modellerin ortak fikri basittir: Bir serinin geleceğini, **yalnızca serinin kendi geçmişini** (geçmiş değerlerini ve geçmişte yapılan tahmin hatalarını) kullanarak tahmin etmek. Dışarıdan ek bir değişkene ihtiyaç duymazlar.

Bölümün yol haritası şöyledir:

1. Yapı taşları: AR, MA, ARMA ve fark alma (I) (7.1)
2. Bu parçaların birleşimi olan ARIMA(p,d,q) ve gecikme operatörüyle yazımı (7.2)
3. Mevsimsel seriler için SARIMA(p,d,q)(P,D,Q)[s] (7.3)
4. Model kurma yöntemi: Box-Jenkins döngüsü (7.4)
5. Durağanlık testleri: ADF ve KPSS (7.5)
6. R ile `AirPassengers` uygulaması (7.6)
7. Aynı analizin Python ile yapılması (7.7)
8. Üstel düzeltme (ETS) ve Theta yöntemi; `AirPassengers` üzerinde SARIMA ile karşılaştırma (7.8)

**Not —** Bu bölüm, Bölüm 3.2'deki temellerin üzerine kurulur: AR(1) denklemi $x_t = \phi x_{t-1} + \varepsilon_t$'nin nasıl okunduğu (alt indisler $t$ ve $t-1$, $\phi$ "fi" ve $\varepsilon$ "epsilon"), bir şokun $\phi$'nin değerine göre sönmesi ya da kalıcı olması, bir denklemin kökünün ne olduğu, gecikme operatörü $B$, birim kök kavramı, ADF testinin hipotezleri ve p-değerinin okunması. Bu konular için önce o bölüme başvurulması önerilir. Burada aynı fikirleri birer adım ileri taşıyacağız.

---

### 7.1. Temel Yapı Taşları: AR, MA, ARMA ve Fark Alma

#### 7.1.1. AR(p): Otoregresif Model

**Açıklama:** Otoregresyonun arkasındaki fikir son derece sezgiseldir: Bir serinin bugünkü değeri, dünkü ve daha önceki değerlerine bağlıdır. Tıpkı bugünkü hava sıcaklığının dünkü sıcaklıktan etkilenmesi gibi.

"Regresyon" kelimesi Latince *regressus* ("geri adım atmak, geri dönmek") kelimesinden gelir. Terimi istatistikte ilk kullananlardan biri Francis Galton'dır. Galton, ebeveynlerin ve çocuklarının boylarını incelerken çok uzun boylu ebeveynlerin çocuklarının da uzun olduğunu, ancak ebeveynleri kadar aşırı uzun olmayıp ortalamaya daha yakın olma eğiliminde olduklarını fark etti. Bu duruma "ortalamaya geri dönüş" (*regression toward the mean*) adını verdi.

Zaman serisinde de benzer bir "geri adım" atarız: Bugünü anlamak için zamanda geriye gidip geçmiş değerlere bakarız. "Oto" ön eki Yunanca "kendi" demektir; yani seri, **kendi geçmişi üzerine** bir regresyon kurar.

**Tanım:** Bir AR(p) modeli, bugünkü değerin geçmişteki $p$ adet değerin ağırlıklı toplamı ile rastgele bir şokun toplamı olduğunu söyler:

```math
\underbrace{x_t}_{\text{bugünkü değer}} = \underbrace{c}_{\text{sabit}} + \underbrace{\overbrace{\phi_1 x_{t-1}}^{\text{dünkü değerin payı}} + \phi_2 x_{t-2} + \dots + \phi_p x_{t-p}}_{\substack{\text{geçmiş }p\text{ değerin}\\\text{ağırlıklı toplamı}}} + \underbrace{\varepsilon_t}_{\text{rastgele şok}}
```

**Ne anlatıyor?** Bugünkü değer; bir sabit, geçmiş $p$ değerin kendi katsayılarıyla çarpılarak elde edilen ağırlıklı toplamı ve öngörülemeyen bir şokun toplamıdır. Katsayılar, geçmişteki her değerin bugüne ne kadar taşındığını belirler.

> **Simge notu:** $`\phi_i`$ *(fi; π "pi" ile karıştırılmamalıdır)*: $`i`$ adım önceki değerin ağırlık katsayısı · $`\varepsilon_t`$ *(epsilon; kümelerdeki "elemanıdır" işareti ∈ değildir)*: $`t`$ anındaki rastgele şok (beyaz gürültü) · $`\dots`$ *(üç nokta)*: aynı kalıp $`p`$'ye kadar sürer

Formülü kelime kelime okuyalım. Alt indisler zamanı gösterir: $x_t$ bu dönemin (örneğin bugünün) değeri, $x_{t-1}$ bir önceki dönemin, $x_{t-2}$ iki dönem öncesinin değeridir; $x_{t-p}$ ise $p$ dönem geriye kadar gidildiğini söyler. $\phi$'nin alt indisi de hangi gecikmeye ait olduğunu gösterir: $\phi_1$ dünkü değerin, $\phi_2$ evvelsi günkü değerin katsayısıdır. **Katsayı**, bir değerin önünde duran ve onu çarpan sayıdır; $0.6\thinspace  x_{t-1}$ ifadesinde katsayı 0.6'dır ve "dünkü değerin %60'ı bugüne taşınır" anlamına gelir. AR(1) denkleminin sözcük sözcük okunuşu ve $\phi$'ye göre şokların nasıl söndüğü Bölüm 3.2'de anlatılmıştı; AR(p) aynı fikri birden çok geçmiş değere genişletir.

Burada:

- $c$ **sabit terimdir** (*constant*): Her dönem değişmeden eklenen bir sayıdır ve serinin hangi seviye etrafında dalgalanacağını belirler. Durağan bir AR(p) sürecinin uzun dönem ortalaması $\mu = c / (1 - \phi_1 - \dots - \phi_p)$'dir. Bunu görmek için serinin hiç şok almadan hep aynı $\mu$ değerinde durduğunu düşünelim: AR(2) için $\mu = c + \phi_1 \mu + \phi_2 \mu$ olur; $\mu$'lü terimleri sola toplayınca $\mu (1 - \phi_1 - \phi_2) = c$ bulunur. Yani $c$ ortalamanın kendisi değil, ortalamayı belirleyen bir sayıdır; $c = 0$ ise seri sıfır etrafında dalgalanır.
- $\varepsilon_t$ modelin açıklayamadığı, öngörülemeyen rastgele şoktur: Bölüm 6.7.1'de tanımladığımız beyaz gürültü, $\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$.

> **Simge notu:** $`\sim`$ *(tilda)*: "… dağılımına sahiptir" · $`\mathrm{WN}(0, \sigma^2)`$: ortalaması 0, varyansı $`\sigma^2`$ *(sigma kare)* olan beyaz gürültü (white noise) · $`\mu`$ *(mü)*: serinin uzun dönem ortalaması · $`\cdot`$: çarpma işareti

**Sayısal örnek (AR(2)).** Bir kafede günlük satılan çay sayısını $x_t$ ile gösterelim ve modelin şu olduğunu varsayalım ($c = 12$, $\phi_1 = 0.6$, $\phi_2 = 0.16$):

```math
x_t = 12 + 0.6\, x_{t-1} + 0.16\, x_{t-2} + \varepsilon_t
```

Önce ortalamayı bulalım: $\mu = 12 / (1 - 0.6 - 0.16) = 12 / 0.24 = 50$ çay. Dün 50, evvelsi gün 40 çay satılmış ($x_{t-1} = 50$, $x_{t-2} = 40$) ve bugün beklenmedik bir kalabalık 3 çaylık sürpriz getirmiş olsun ($\varepsilon_t = 3$):

```math
\begin{aligned}
x_t &= 12 + 0.6 \cdot 50 + 0.16 \cdot 40 + 3 \\
    &= 12 + 30 + 6.4 + 3 \\
    &= 51.4
\end{aligned}
```

Ertesi gün her şey bir adım kayar: "dün" artık 51.4, "evvelsi gün" 50 olur. Sürpriz bu kez −1 ise $x_{t+1} = 12 + 0.6 \cdot 51.4 + 0.16 \cdot 50 - 1 = 12 + 30.84 + 8 - 1 = 49.84$ bulunur. Seri 50'lik ortalamanın çevresinde dolaşır: Şoklar onu biraz yukarı ya da aşağı iter, geçmiş değerlerin ağırlıklı toplamı ise ortalamaya doğru geri çeker. Tahmin yaparken geleceğin şokunu bilemeyiz; onun yerine şokların ortalaması olan 0'ı koyarız. Bugün için tahmin $12 + 30 + 6.4 = 48.4$ olurdu; gerçekleşen 51.4 ile arasındaki 3'lük fark tam da o günün şokudur.

$p$ değeri modelin "hafızasının" ne kadar geriye gittiğini belirtir. Örneğin AR(1) modeli yalnızca bir önceki değerin bugünü etkilediğini varsayar: $x_t = c + \phi_1 x_{t-1} + \varepsilon_t$.

**Not —** AR(1) sürecinin durağan olması için $\lvert \phi_1 \rvert < 1$ olmalıdır ($\lvert \cdot \rvert$ **mutlak değer**dir: sayının işaretini atıp yalnızca büyüklüğüne bakar, örneğin $\lvert -0.7 \rvert = 0.7$). $\phi_1 = 1$ olursa süreç rastgele yürüyüşe (birim kök, Bölüm 3.2) dönüşür ve şoklar hiç sönmez. AR(p) için koşul tek tek katsayılara bakarak değil, 7.2'de göreceğimiz karakteristik denklemin kökleriyle kontrol edilir.

#### 7.1.2. MA(q): Hareketli Ortalama Modeli

**Açıklama:** Buradaki "hareketli ortalama", `decompose()`'un trendi bulmak için kullandığı düzleştirmeyle aynı şey değildir: Burada gözlemlerin değil, geçmiş şokların ağırlıklı toplamı alınır (bkz. 6.7.1). MA modeli, önceki adımlardaki öngörülemeyen "şokları" bugünkü değeri açıklamak için kullanır.

**Günlük hayattan bir örnek:** Bir vadide bağıran birini düşünelim. O an duyulan ses bağırışın kendisidir ($\varepsilon_t$); bir an sonra bağırışın yankısı daha zayıf olarak geri döner ($\theta_1 \varepsilon_{t-1}$), ardından sessizlik olur. Her an duyulan ses, o anki bağırış ile bir önceki bağırışın yankısının toplamıdır; MA(1) budur. MA(q)'da yankı $q$ kez döner, sonra tamamen kaybolur. Benzetmenin sınırı: $\theta$ negatif de olabilir; ters işaretli bir yankının gündelik deneyimde karşılığı yoktur, ama modelde şokun ertesi dönemde ters yönde iz bırakması demektir.

**Tanım:** Bir MA(q) süreci, bugünkü değerin serinin ortalaması, bugünkü şok ve geçmişteki $q$ adet şokun ağırlıklı toplamından oluştuğunu söyler:

```math
\underbrace{x_t}_{\text{bugünkü değer}} = \underbrace{\mu}_{\text{ortalama}} + \underbrace{\varepsilon_t}_{\text{bugünkü şok}} + \underbrace{\overbrace{\theta_1 \varepsilon_{t-1}}^{\text{dünkü şokun yankısı}} + \theta_2 \varepsilon_{t-2} + \dots + \theta_q \varepsilon_{t-q}}_{\substack{\text{geçmiş }q\text{ şokun}\\\text{ağırlıklı toplamı}}}
```

**Ne anlatıyor?** Bugünkü değer, ortalamanın üstüne bugünkü şokun ve son $q$ şokun zayıflamış yankılarının eklenmesiyle oluşur. $q$ dönemden daha eski şoklar bugünkü değeri etkilemez.

> **Simge notu:** $`\mu`$ *(mü)*: serinin ortalaması · $`\theta_j`$ *(teta)*: $`j`$ adım önceki şokun ağırlık katsayısı · $`\varepsilon_{t-1}`$: bir önceki dönemin şoku

AR modelindeki $c$'den farklı olarak MA modelinde sabit terim doğrudan serinin ortalamasıdır ($\mu$), çünkü denklemde geçmiş değerler yoktur.

**Not —** Katsayıların işaret kuralı yazılımdan yazılıma değişebilir. R (`arima`, `auto.arima`) ve Python `statsmodels`/`pmdarima`, MA kısmını yukarıdaki gibi **artı** işaretiyle ($+ \theta_1 \varepsilon_{t-1}$) yazar. Bazı ders kitapları ise eksi işareti kullanır; sonuçlar karşılaştırılırken buna dikkat edilmelidir.

**Sayısal örnek (MA(1)).** $\mu = 100$ ve $\theta_1 = 0.6$ olan bir MA(1) süreci düşünelim: $x_t = 100 + \varepsilon_t + 0.6\thinspace  \varepsilon_{t-1}$. Şokun nasıl girip çıktığını net görmek için yalnızca 1. dönemde $\varepsilon_1 = 10$ büyüklüğünde bir şok olduğunu, diğer bütün dönemlerde şokun 0 olduğunu varsayalım:

| Dönem $`t`$ | Bugünkü şok $`\varepsilon_t`$ | Dünkü şok $`\varepsilon_{t-1}`$ | Hesap | $`x_t`$ |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | $`100 + 0 + 0.6 \cdot 0`$ | 100 |
| 1 | 10 | 0 | $`100 + 10 + 0.6 \cdot 0`$ | 110 |
| 2 | 0 | 10 | $`100 + 0 + 0.6 \cdot 10`$ | 106 |
| 3 | 0 | 0 | $`100 + 0 + 0.6 \cdot 0`$ | 100 |

Şok 1. dönemde tam gücüyle girer (+10), 2. dönemde yalnızca %60'ı kalır (+6), 3. dönemde model onu artık "hatırlamaz" ve seri ortalamasına döner. Yankı benzetmesiyle: Bağırışın yankısı yalnızca bir kez döner; ikinci yankı yoktur. Aynı şok, ortalaması 100 olan bir AR(1) modelinde ($\phi_1 = 0.6$) 110, 106, 103.6, 102.16, … biçiminde **yavaş yavaş** söner ama hiçbir zaman tam sıfırlanmaz, çünkü her değer bir öncekinden pay alır. MA ile AR arasındaki temel fark budur.

$\theta$ katsayıları geçmiş şokların bugünkü değeri ne kadar etkilediğini belirler. İsim de bu formülden gelir: Model, geçmiş şokların "kayan" bir ağırlıklı toplamını kullanır. MA süreci bu yüzden **kısa hafızalıdır** ve ACF'si $q$ gecikmeden sonra kesilir (Bölüm 6.7.2).

> **Hemen deneyin 7.1.2 · MA(1) üret, katsayıyı geri bul** (Python, ~5 dk)
>
> $\mu = 100$, $\theta_1 = 0.6$ olan bir MA(1)'den 500 gözlem üretip `statsmodels` ile modeli veriden kestirelim:
>
> ```python
> import numpy as np
> from statsmodels.tsa.arima.model import ARIMA
>
> np.random.seed(1)
> eps = np.random.normal(0, 1, 501)          # 501 rastgele şok (beyaz gürültü)
> x = 100 + eps[1:] + 0.6 * eps[:-1]         # x_t = 100 + e_t + 0.6 e_(t-1)
> model = ARIMA(x, order=(0, 0, 1)).fit()    # order = (p, d, q): saf MA(1)
> print(model.params.round(3))               # sabit, ma.L1, sigma2
> ```
>
> `eps[1:]` "bugünkü şoklar", `eps[:-1]` bir adım geride kalan "dünkü şoklar"dır; ikisi 500'er elemanlıdır. `.fit()` katsayıları en çok olabilirlik yöntemiyle kestirir (Bölüm 7.4).
>
> **Ne göreceksiniz:** `[100.077   0.574   0.975]`. Sabit (ortalama) 100.08, $\hat{\theta}_ 1 = 0.574$, şok varyansı 0.975: Gerçek değerler olan 100, 0.6 ve 1'e yakındır. Sapma, sonlu örneklemin doğal sonucudur.
>
> **Şimdi siz:** `x = ...` satırındaki `+ 0.6` ifadesini `- 0.6` yapıp tekrar çalıştırın (`np.random.seed(1)` satırı da yeniden çalışmalı). `ma.L1` hangi işaretle gelir?
>
> <details><summary>Cevap</summary>
>
> `[100.019  -0.622   0.982]`: $\hat{\theta}_ 1 = -0.622$. `statsmodels` MA kısmını artı işaretiyle yazdığı için ($x_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1}$, 7.1.2'deki not) eksiyle ürettiğimiz katsayıyı eksi olarak bulur.
>
> </details>

#### 7.1.3. ARMA(p, q): İkisinin Birleşimi

**Açıklama:** Gerçek serilerde çoğu zaman hem "geçmiş değerlerin" hem de "geçmiş şokların" etkisi bir aradadır. ARMA modeli bu iki fikri tek denklemde birleştirir. Böylece saf AR ya da saf MA ile çok sayıda terim gerektirecek bir yapı, az sayıda parametreyle ifade edilebilir.

**Tanım:**

```math
x_t = c + \underbrace{\phi_1 x_{t-1} + \dots + \phi_p x_{t-p}}_{\substack{\text{AR kısmı:}\\\text{geçmiş değerler}}} + \overbrace{\varepsilon_t}^{\text{bugünkü şok}} + \underbrace{\theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}}_{\substack{\text{MA kısmı:}\\\text{geçmiş şoklar}}}
```

**Ne anlatıyor?** Bugünkü değer hem geçmiş $p$ değerin hem de bugünkü ve geçmiş $q$ şokun ağırlıklı toplamıyla açıklanır; AR ve MA yapıları tek denklemde birleşir.

Bu denklem, 7.1.1'deki AR(p) ile 7.1.2'deki MA(q) denklemlerinin sağ taraflarının yan yana yazılmasıdır: önce geçmiş **değerler** ($\phi$'li terimler), sonra bugünkü ve geçmiş **şoklar** ($\varepsilon$ ve $\theta$'lı terimler).

**Sayısal örnek (ARMA(1,1)).** Basitlik için $c = 0$ (seri sıfır etrafında dalgalanıyor), $\phi_1 = 0.5$ ve $\theta_1 = 0.4$ alalım: $x_t = 0.5\thinspace  x_{t-1} + \varepsilon_t + 0.4\thinspace  \varepsilon_{t-1}$. Yine yalnızca 1. dönemde $\varepsilon_1 = 10$ şoku olsun ve seri 0'dan başlasın:

```math
\begin{aligned}
x_1 &= 0.5 \cdot 0 + 10 + 0.4 \cdot 0 = 10 \\
x_2 &= 0.5 \cdot 10 + 0 + 0.4 \cdot 10 = 5 + 4 = 9 \\
x_3 &= 0.5 \cdot 9 + 0 + 0.4 \cdot 0 = 4.5 \\
x_4 &= 0.5 \cdot 4.5 + 0 + 0.4 \cdot 0 = 2.25
\end{aligned}
```

Saf AR(1) ($\phi_1 = 0.5$) aynı şoka 10, 5, 2.5, 1.25 diye tepki verirdi. ARMA(1,1)'de MA kısmı ikinci dönemde şoka bir kez daha +4'lük "ek itme" yapar; sonra devreden çıkar ve AR kısmı etkiyi her dönem yarıya indirerek söndürür. Saf bir AR modeliyle bu "önce ek itme, sonra yarılanma" desenini kurmak için çok sayıda gecikme gerekir; ARMA bunu iki katsayıyla yapar. Az parametreli (tutumlu) model kurmanın anlamı budur.

ARMA modeli **durağan** bir seri varsayar. ACF ve PACF grafiklerinin ikisi de keskin bir kesilme göstermeden yavaşça sönümleniyorsa ARMA yapısından şüphelenilir (Bölüm 6.7.5).

> **Hemen deneyin 7.1.3 · Şok tepkisini R'a hesaplatın** (R, ~2 dk)
>
> `ARMAtoMA()` bir ARMA modelinin, 1 birimlik bir şoka sonraki dönemlerde verdiği tepkiyi hesaplar.
>
> ```r
> ARMAtoMA(ar = 0.5, ma = 0.4, lag.max = 5)   # şoktan 1, 2, ..., 5 dönem sonraki tepki
> ```
>
> **Ne göreceksiniz:** `[1] 0.90000 0.45000 0.22500 0.11250 0.05625`. Şok 10 birim olduğunda bu sayılar 10 ile çarpılır: 9, 4.5, 2.25, … Metindeki elle hesaplanan $x_2 = 9$, $x_3 = 4.5$, $x_4 = 2.25$ değerlerinin aynısı.
>
> **Şimdi siz:** Aynı fonksiyonu yalnızca `ar = 0.5` ile, sonra yalnızca `ma = 0.4` ile çalıştırın. ARMA'daki 0.9, iki parçadan nasıl oluşuyor?
>
> <details><summary>Cevap</summary>
>
> Saf AR(1): `0.50000 0.25000 0.12500 0.06250 0.03125`. Saf MA(1): `0.4 0.0 0.0 0.0 0.0`. İlk dönemdeki 0.9 = 0.5 (AR'ın taşıdığı pay) + 0.4 (MA'nın "ek itmesi"); sonrasında MA devreden çıkar ve AR her dönem yarıya indirir.
>
> </details>

#### 7.1.4. I: Entegrasyon ve Fark Alma

**Açıklama:** Fark operatörü $\nabla$ ve fark alma Bölüm 2.1'de tanıtıldı; burada konuya model açısından bakıyoruz. Birçok seri, özellikle ekonomi ve finansta, durağan değildir: Yıllar içinde büyüyen bir şirketin satışlarının ortalaması sabit kalmaz. AR ve MA modelleri ise serinin istatistiksel özelliklerinin zamanla değişmediğini, yani **durağan** olduğunu varsayar. Sürekli yer değiştiren bir hedefi vurmaya çalışmak çok daha zordur. Çözüm, serinin kendisi yerine **değişimini** modellemektir: Bugünkü satış yerine bugünkü satış ile dünkü satış arasındaki farkı tahmin ederiz.

**Tanım:** Birinci fark şöyle yazılır:

```math
\underbrace{w_t}_{\substack{\text{farkı alınmış}\\\text{seri}}} = \underbrace{\nabla x_t}_{\text{birinci fark}} = \overbrace{x_t - x_{t-1}}^{\text{bugün eksi dün}}
```

**Ne anlatıyor?** Birinci fark, her dönemin değerinden bir önceki dönemin değerinin çıkarılmasıyla bulunur; böylece seviye yerine dönemden döneme değişim incelenir.

> **Simge notu:** $`\nabla`$ *(nabla)*: fark operatörü, $`\nabla x_t = x_t - x_{t-1}`$ · $`\nabla^2`$ *(nabla kare)*: farkın bir kez daha farkı · $`\nabla_s`$ *(nabla s)*: $`s`$ adımlık mevsimsel fark

**Sayısal örnek.** Bir mağazanın beş aylık satışları 100, 104, 110, 113, 120 ise birinci farklar 4, 6, 3, 7'dir. Seviye 100'den 120'ye tırmanırken farklar yerinde sayar: Trend gitmiş, geriye "aylık artış" kalmıştır. İlk ayın "dünü" olmadığı için fark alınca bir gözlem kaybedilir.

**Günlük hayattan bir örnek:** Bir banka hesabının bakiyesi yıllar içinde maaşlar ve harcamalarla kayar; bakiyeye bakarak "normal bir ay" nedir söylemek zordur. Hesap hareketlerine, yani bakiyenin her ay ne kadar değiştiğine bakıldığında benzer büyüklükte artı ve eksiler görülür. Fark almak, bakiye yerine hareket dökümüne bakmaktır. Benzetmenin sınırı: Hareketlerin kendisi de büyüyebilir (örneğin her yıl artan maaş); o zaman bir fark daha, yani $d = 2$ gerekir.

Farkı alınmış seri hâlâ durağan değilse işlem bir kez daha uygulanır: $\nabla^2 x_t = w_t - w_{t-1}$. Örneğimizde ikinci farklar $6 - 4 = 2$, $3 - 6 = -3$ ve $7 - 3 = 4$ olur. Seriyi durağanlaştırmak için kaç kez fark alındığı, ARIMA(p,d,q) modelindeki **d** parametresidir. Pratikte $d$ genellikle 0, 1 ya da en fazla 2'dir.

**Aşırı fark alma.** Gereğinden fazla fark almak da zararlıdır. Zaten durağan bir seriye fark alınırsa seri iyileşmez, bozulur: Varyansı 1 olan bir beyaz gürültünün farkı alınınca varyans 2'ye çıkar ve lag-1 otokorelasyonu yaklaşık −0.5 olur. Seriye aslında olmayan bir zikzak eklenmiştir. Belirtileri şunlardır: Farkı alınmış serinin ACF'sinde lag-1'de −0.5'e yakın ya da daha negatif bir çubuk ve modelde −1'e yakın bir MA katsayısı (7.2'deki tersinirlik sınırı). Banka benzetmesiyle: Hareket dökümünün bir kez daha farkını almak "bu ayki harcama eksi geçen ayki harcama"ya bakmaktır; geçen ay tatile çıkılmışsa bu ay sahte bir "tasarruf" görünür.

Mevsimsel seriler için ayrıca **mevsimsel fark** alınır: $\nabla_s x_t = x_t - x_{t-s}$. Aylık veride ($s = 12$) bu, "bu Ocak eksi geçen Ocak" demektir ve yıllık tekrar eden deseni siler. Örneğin geçen Ocak 200, bu Ocak 230 yolcu taşındıysa $\nabla_{12} x_t = 230 - 200 = 30$'dur.

"Integrated" (entegre) terimi bu işlemin tersini ifade eder. Model farkı alınmış seri için tahmin ürettikten sonra, bu tahminlerin art arda toplanarak (entegre edilerek) orijinal ölçeğe geri döndürülmesi gerekir. Örneğimizde ilk değer (100) ve farklar (4, 6, 3, 7) biliniyorsa seri art arda toplamayla (**kümülatif toplam**) geri kurulur: 100, $100 + 4 = 104$, $104 + 6 = 110$, $110 + 3 = 113$, $113 + 7 = 120$. Tahminde de aynısı olur: Model gelecek ay için farkı +5, sonraki ay için +6 tahmin ederse seviye tahminleri $120 + 5 = 125$ ve $125 + 6 = 131$ olur. Özetle: Fark alarak seriyi analiz edilebilir hâle getiririz, modelleriz, sonra sonucu orijinal bağlamına entegre ederiz. R ve Python fonksiyonları bu geri dönüşü bizim yerimize otomatik yapar. Banka benzetmesiyle entegrasyon, başlangıç bakiyesine hareketleri art arda ekleyip bugünkü bakiyeyi bulmaktır.

Şekil 7.1, `AirPassengers` serisi üzerinde bu adımları göstermektedir. Log dönüşümü dalgaların giderek büyümesini (artan varyansı) dengeler; ardından alınan mevsimsel ve normal fark, trendi ve yıllık deseni silerek sıfır çevresinde dalgalanan durağan bir seri bırakır. Log dönüşümünün işe yarama nedeni, logaritmanın çarpmayı toplamaya çevirmesidir: Log serinin farkı yaklaşık olarak **yüzde değişimi** verir. Örneğin 100'den 110'a çıkış için $\log(110) - \log(100) = 0.0953$, yani yaklaşık %10'dur; 1000'den 1100'e çıkışta da aynı 0.0953 bulunur. Böylece yüksek seviyelerdeki büyük dalgalanmalar ile düşük seviyelerdeki küçük dalgalanmalar aynı ölçeğe gelir. (Bu bölümde $\log$, R'daki `log()` ve Python'daki `np.log()` gibi doğal logaritmadır.)

![Fark alma ile durağanlaştırma](../images/ch07_fark_alma.svg)

*Şekil 7.1 — `AirPassengers` serisinin durağanlaştırılması: (a) orijinal seri, (b) log dönüşümü, (c) log serinin 12 aylık ve ardından 1 aylık farkı ($`\nabla \nabla_{12} \log x_t`$).*

> **Hemen deneyin 7.1.4 · Fark al, geri topla** (R, ~5 dk)
>
> ```r
> satis <- c(100, 104, 110, 113, 120)
> diff(satis)                       # birinci fark
> diff(satis, differences = 2)      # ikinci fark (farkın farkı)
> diffinv(diff(satis), xi = 100)    # farkları art arda topla: seriyi geri kur
> ```
>
> `diffinv()` fark almanın tersidir ("entegrasyon"): `xi` başlangıç değeridir, farklar bunun üzerine art arda eklenir.
>
> **Ne göreceksiniz:** `[1] 4 6 3 7`, `[1]  2 -3  4` ve `[1] 100 104 110 113 120`: metinde elle bulduğumuz farklar ve geri kurulan seri.
>
> **Şimdi siz:** (1) Model gelecek iki ay için farkı +5 ve +6 tahmin ediyor. Seviye tahminlerini `diffinv()` ile bulun. (2) `diff(AirPassengers, lag = 12)[7]` neyi hesaplar ve kaç verir? (3) Aşırı fark almayı görün: `set.seed(1)` ve `e <- rnorm(500)` ile zaten durağan bir beyaz gürültü üretin; `var(e)`, `var(diff(e))` ve `acf(diff(e), lag.max = 3, plot = FALSE)` ne gösterir?
>
> <details><summary>Cevap</summary>
>
> (1) `diffinv(c(5, 6), xi = 120)` → `120 125 131`; ilk eleman başlangıç değeridir, tahminler 125 ve 131'dir. (2) Mevsimsel farkın 7. değeri: Temmuz 1950 − Temmuz 1949 = 170 − 148 = `22`. (3) Varyans `1.023999`'dan `2.105146`'ya çıkar; lag-1 otokorelasyonu `-0.511`'dir (teorik değer −0.5), sonraki gecikmeler sıfıra yakındır. Farkı alınmış serideki zikzak veriden değil, gereksiz fark işleminden gelir.
>
> </details>

---

### 7.2. ARIMA(p, d, q) Modeli ve Gecikme Operatörüyle Yazım

Üç bileşen bir araya gelerek **ARIMA(p, d, q)** (AutoRegressive Integrated Moving Average) modelini oluşturur:

| Parametre | Bileşen | Anlamı |
| --- | --- | --- |
| **p** | AR | Kaç geçmiş **değerin** kullanılacağı |
| **d** | I | Seriyi durağanlaştırmak için kaç kez **fark** alındığı |
| **q** | MA | Kaç geçmiş **hatanın (şokun)** kullanılacağı |

Örneğin ARIMA(1,1,0), "seriyi bir kez farkla, farkı alınmış seriye AR(1) uygula" demektir. ARIMA(0,0,0) ise saf beyaz gürültüdür; ARIMA(0,1,0) rastgele yürüyüştür.

**Gecikme operatörüyle yazım.** ARIMA denklemleri açık hâliyle uzun ve okunaksızdır. Bölüm 2.1'de tanıttığımız gecikme (backshift) operatörü $B$ ile bu denklemleri çok kısa yazabiliriz. Hatırlatma: $B$ bir sayı değil, "önündeki seriyi bir dönem geri kaydır" komutudur; $B x_t = x_{t-1}$ ve genel olarak $B^k x_t = x_{t-k}$'dir (bazı kaynaklarda $L$; açıklamalı gösterimi ve okunuşu Bölüm 6.2.2'de).

> **Simge notu:** $`B`$ *(be)*: gecikme operatörü, seriyi bir adım geriye kaydırır · $`B^k`$ *(be üzeri k)*: $`k`$ adım geri kaydırma

Önce AR ve MA kısımlarını birer **polinom** (Bölüm 3.2.3) olarak tanımlarız. Bilinmeyenin yerinde $B$ durduğunda ortaya bir "gecikme polinomu" çıkar:

```math
\begin{aligned}
\underbrace{\phi(B)}_{\text{AR polinomu}} &= 1 - \underbrace{\phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p}_{\substack{p\text{ geçmiş değerin}\\\text{katsayıları}}} \\
\underbrace{\theta(B)}_{\text{MA polinomu}} &= 1 + \underbrace{\theta_1 B + \theta_2 B^2 + \dots + \theta_q B^q}_{\substack{q\text{ geçmiş şokun}\\\text{katsayıları}}}
\end{aligned}
```

**Ne anlatıyor?** AR polinomu geçmiş değerlerin, MA polinomu ise geçmiş şokların katsayılarını gecikme operatörü cinsinden tek bir ifadede toplar. Uzun denklemler böylece kısa bir çarpım biçiminde yazılabilir.

$\phi(B)$ "fi B" diye okunur ve "$`\phi`$ katsayılarıyla kurulmuş, $B$ cinsinden polinom" demektir; $\phi$ ile $B$'nin çarpımı değil, bir addır. Bir gecikme polinomunu seriye uygulamak, parantezdeki her terimi seriye ayrı ayrı uygulayıp sonuçları toplamak demektir. 7.1.1'deki AR(2) örneğiyle ($\phi_1 = 0.6$, $\phi_2 = 0.16$):

```math
(1 - 0.6B - 0.16B^2)\, x_t = x_t - 0.6\, x_{t-1} - 0.16\, x_{t-2}
```

Bunu $12 + \varepsilon_t$'ye eşitleyip geçmiş değerleri eşitliğin sağına taşırsak 7.1.1'deki $x_t = 12 + 0.6\thinspace  x_{t-1} + 0.16\thinspace  x_{t-2} + \varepsilon_t$ denklemini geri buluruz. Yani polinom yazımı yeni bir model değil, aynı modelin kısaltmasıdır. Bu polinomlarla modeller şu biçimi alır:

```math
\begin{aligned}
\text{AR}(p):&\quad \phi(B)\, x_t = c + \varepsilon_t \\
\text{MA}(q):&\quad x_t = \mu + \theta(B)\, \varepsilon_t \\
\text{ARMA}(p,q):&\quad \phi(B)\, x_t = c + \theta(B)\, \varepsilon_t \\
\text{ARIMA}(p,d,q):&\quad \underbrace{\phi(B)}_{\text{AR kısmı}}\, \underbrace{(1 - B)^d}_{d\text{ kez fark al}}\, x_t = c + \underbrace{\theta(B)}_{\text{MA kısmı}}\, \varepsilon_t
\end{aligned}
```

**Ne anlatıyor?** Dört model aynı kalıpla yazılır: AR kısmı seriye, MA kısmı şoklara uygulanır. ARIMA'da bunlara ek olarak seriye önce $d$ kez fark uygulanır.

Son satırı okuyalım: $(1 - B)^d$ çarpanı seriye $d$ kez fark uygular (çünkü $(1 - B)x_t = x_t - x_{t-1} = \nabla x_t$). Ortaya çıkan durağan seri, $\phi(B)$ ile AR yapısına, $\theta(B)$ ile MA yapısına bağlanır. Yani ARIMA, "farkı alınmış serinin ARMA modeli"dir.

Fark alındığında sabit terimin anlamı da değişir. $d = 0$ iken $c$ serinin seviyesini belirler (7.1.1). $d = 1$ iken ise farkı alınmış serinin, yani dönemden döneme değişimin ortalamasını belirler: $c \neq 0$ seriye sabit eğimli bir trend ekler. Bu terime **kayma** (*drift*) denir. $d = 2$ iken sabit terim ikinci dereceden (eğri) bir trend anlamına gelir; bu, geleceğe taşınması riskli bir varsayımdır. R'daki `auto.arima()` bu yüzden toplam fark sayısı $d + D$ 2 ya da daha büyükse sabit terim eklemez; $d + D = 1$ ise `drift` adlı bir terimi aday olarak dener.

**Örnek: ARIMA(1,1,1)'in açık hâli.** Bu modelde AR(1) kısmı $(1 - \phi_1 B)$, fark çarpanı $(1 - B)$, MA(1) kısmı $(1 + \theta_1 B)$'dir. Önce $(1 - B)x_t = x_t - x_{t-1}$ ile bu ayın değişimi bulunur, AR(1) kısmı bu değişim serisine uygulanır; sağ taraf $\varepsilon_t + \theta_1 \varepsilon_{t-1}$ olur. $\phi_1$'li terim sağa taşınınca sonuç şudur:

```math
\begin{aligned}
(1 - \phi_1 B)(1 - B)\, x_t &= (1 + \theta_1 B)\, \varepsilon_t \\[6pt]
\underbrace{x_t - x_{t-1}}_{\text{bu ayki değişim}} &= \underbrace{\phi_1 (x_{t-1} - x_{t-2})}_{\substack{\text{geçen ayki değişimin}\\\text{payı (AR)}}} + \overbrace{\varepsilon_t}^{\text{bu ayın şoku}} + \underbrace{\theta_1 \varepsilon_{t-1}}_{\substack{\text{geçen ayın şokunun}\\\text{yankısı (MA)}}}
\end{aligned}
```

**Ne anlatıyor?** Kısaltılmış yazım açıldığında ARIMA(1,1,1)'in, seviyeler yerine değişimler üzerine kurulmuş bir ARMA(1,1) modeli olduğu görülür.

Yani bu ayki **değişim**, geçen ayki değişim, bu ayın şoku ve geçen ayın şoku ile açıklanır. Sayılarla: $\phi_1 = 0.5$, $\theta_1 = 0.3$, son iki ayın değerleri $x_{t-2} = 100$ ve $x_{t-1} = 104$, geçen ayın şoku $\varepsilon_{t-1} = 2$ ve bu ayın şoku $\varepsilon_t = 1$ olsun. Bu ayki değişim $0.5 \cdot (104 - 100) + 1 + 0.3 \cdot 2 = 2 + 1 + 0.6 = 3.6$, bu ayın değeri de $104 + 3.6 = 107.6$ olur.

**Not —** R ve Python, MA polinomunu yukarıdaki gibi artı işaretiyle ($1 + \theta_1 B$) yazar; bazı ders kitaplarındaki eksi işaretli yazımla karşılaştırılırken dikkat edilmelidir (7.1.2'deki not).

**Durağanlık koşulu: karakteristik denklem ve kökler.** Bölüm 3.2'de AR(1) için şunu gördük: $B$ komutunun yerine sıradan bir bilinmeyen $z$ koyup $1 - \phi z = 0$ denklemini (**karakteristik denklem**) kurarız; kökü $z = 1/\phi$'dir ve bu kök 1'e eşitse "birim kök" vardır. Aynı yöntem her AR(p) için işler: $\phi(B)$ polinomunda $B$ yerine $z$ yazıp sıfıra eşitleriz:

```math
\underbrace{1 - \phi_1 z - \phi_2 z^2 - \dots - \phi_p z^p}_{\substack{\phi(B)\text{ polinomunda}\\B\text{ yerine }z}} = 0
```

**Ne anlatıyor?** AR(p) modelinin karakteristik denklemi, AR polinomunda gecikme operatörü yerine $z$ yazılıp sıfıra eşitlenerek kurulur; köklerin konumu modelin durağan olup olmadığını belirler.

$p$. dereceden bir polinom denkleminin $p$ tane kökü vardır (bazıları aynı değer olabilir). Kural, Bölüm 3.2'de AR(1) için gördüğümüzle aynıdır:

> **Durağanlık kuralı:** AR kısmının karakteristik denkleminin **bütün kökleri mutlak değerce 1'den büyükse** ($`\lvert z \rvert > 1`$, yani "birim çemberin dışında") süreç durağandır. Köklerden biri tam 1 ise birim kök vardır; mutlak değeri 1'den küçük bir kök patlayan bir seri demektir.

Kökler sıradan (gerçel) sayılar olduğunda kural "kök, sayı doğrusunda −1 ile 1 arasında (uçlar dahil) olmamalı" demektir; karmaşık köklerde bu sınırın neden bir çembere dönüştüğü Bölüm 3.2'de (Şekil 3.6) gösterilmişti. AR(p)'de yeni olan, **tek bir kökün değil, bütün köklerin** bu koşulu sağlaması gerektiğidir.

**Örnek (durağan AR(2)).** 7.1.1'deki modelin ($\phi_1 = 0.6$, $\phi_2 = 0.16$) karakteristik denklemi $1 - 0.6z - 0.16z^2 = 0$'dır. Sol taraf $(1 - 0.8z)(1 + 0.2z)$ biçiminde çarpanlarına ayrılır; kökler parantezleri tek tek sıfır yapan değerlerdir:

```math
\begin{aligned}
1 - 0.8z = 0 &\;\Rightarrow\; z = 1 / 0.8 = 1.25 \\
1 + 0.2z = 0 &\;\Rightarrow\; z = -1 / 0.2 = -5
\end{aligned}
```

İki kökün de mutlak değeri 1'den büyüktür (1.25 ve 5), dolayısıyla model durağandır. 7.1.1'de serinin 50'lik ortalama çevresinde dolaştığını görmemiz bununla tutarlıdır.

**Birim kök ve fark.** Bölüm 3.2.3'teki birim köklü AR(2) örneğinde ($1 - 1.5z + 0.5z^2 = (1 - z)(1 - 0.5z) = 0$, kökler 1 ve 2) aynı çarpanlara ayırma $B$ ile yazılınca model $(1 - 0.5B)(1 - B)\, x_t = \varepsilon_t$ olur: $(1 - B)$ çarpanı bir kez fark alır, geriye $\phi_1 = 0.5$ olan durağan bir AR(1) kalır; yani bu "AR(2)" aslında bir **ARIMA(1,1,0)** modelidir. Genel ders şudur: Karakteristik denklemdeki her birim kök, modelden bir $(1 - B)$ çarpanı olarak ayrılır ve bir fark alma işlemine karşılık gelir. ARIMA'daki $d$, serideki birim kök sayısıdır.

Şekil 7.2 bu iki AR(2) modelinin ve üç AR(1) modelinin köklerini sayı doğrusu üzerinde gösterir.

![Karakteristik kökler ve birim çember](../images/ch07_karakteristik_kokler.svg)

*Şekil 7.2 — Karakteristik denklemin kökleri. Gri bant, mutlak değeri 1'den küçük sayıları (birim çemberin içini) gösterir. Durağan modellerin bütün kökleri bandın dışındadır; birim köklü modellerde bir kök tam 1'dedir; patlayan AR(1)'in ($`\phi = 1.2`$) kökü ($`z = 1/1.2 \approx 0.83`$) bandın içindedir.*

Kökleri elle bulmak yalnızca bu tür basit örneklerde kolaydır; yüksek dereceli polinomlarda Bölüm 3.2.3'te tanıdığımız `Mod(polyroot(...))` kullanılır (örneğin `Mod(polyroot(c(1, -0.6, -0.16)))` → `1.25 5.00`).

**Not —** MA kısmı için de benzer bir koşul vardır: $\theta(z) = 0$ denkleminin kökleri de birim çemberin dışında olmalıdır. Bu koşula **tersinirlik** (*invertibility*) denir ve geçmiş şokların gözlenen veriden tek bir şekilde geri hesaplanabilmesini sağlar. R ve Python model kurarken bu koşulları kendileri denetler.

Koşul neden gerekli? $\theta = 0.5$ ve $\theta = 2$ olan iki MA(1)'in ACF'si aynıdır: $\rho_1 = \theta / (1 + \theta^2) = 0.4$. Veri ikisini ayırt edemez; yazılım tersinir olanı (0.5) seçer.

> **Hemen deneyin 7.2 · AR(1) üret, $\phi$'yi ve kökü geri bul** (R, ~5 dk)
>
> ```r
> library(forecast)
> set.seed(2024)
> x <- arima.sim(model = list(ar = 0.7), n = 300)             # gerçek phi = 0.7
> fit <- Arima(x, order = c(1, 0, 0), include.mean = FALSE)   # AR(1) kur, sabit yok
> fit
> polyroot(c(1, -coef(fit)["ar1"]))                           # 1 - phi z = 0 denkleminin kökü
> ```
>
> `Arima()` (`forecast` paketi) derecelerini bizim verdiğimiz bir ARIMA modelini kurar; `order = c(p, d, q)`. `include.mean = FALSE` sabit terim eklememesini söyler (seri sıfır çevresinde üretildi). `coef(fit)["ar1"]` modelden `ar1` adlı katsayıyı çeker.
>
> **Ne göreceksiniz:** Çıktıda `ARIMA(1,0,0) with zero mean`, `ar1 = 0.6907`, `s.e. 0.0419` (`s.e.`, *standard error*: standart hata, yani katsayı kestiriminin belirsizliği; tanımı 7.5'te). Ardından `[1] 1.447763+0i`: Kestirilen modelin karakteristik kökü $1/0.6907 \approx 1.45$'tir; 1'den büyük olduğu için model durağandır (gerçek kök $1/0.7 \approx 1.43$).
>
> **Şimdi siz:** (1) `n = 3000` ile tekrarlayın (önce `set.seed(2024)`). Standart hata ne kadar küçülür? (2) `Mod(polyroot(c(1, -0.5, -0.5)))` ile $x_t = 0.5\thinspace x_{t-1} + 0.5\thinspace x_{t-2} + \varepsilon_t$ modelini denetleyin. Durağan mı?
>
> <details><summary>Cevap</summary>
>
> (1) `ar1 = 0.6858`, `s.e. 0.0133`: Veri 10 katına çıkınca standart hata yaklaşık $\sqrt{10} \approx 3.2$ kat küçülür. 0.7 değeri hâlâ "kestirim ± 2 s.e." aralığındadır. (2) `[1] 1 2`: Bir kök tam 1, yani birim kök var; model durağan değildir ve bir kez fark alınınca ARIMA(1,1,0) olur.
>
> </details>

---

### 7.3. Mevsimsel ARIMA: SARIMA(p, d, q)(P, D, Q)[s]

**Açıklama:** `AirPassengers` gibi aylık verilerde iki tür ilişki vardır: Ardışık aylar arasındaki ilişki (Şubat, Ocak'a benzer) ve **aynı mevsimler** arasındaki ilişki (bu Temmuz, geçen Temmuz'a benzer). Normal ARIMA yalnızca ilkini modeller. **SARIMA** (Seasonal ARIMA), aynı AR/I/MA fikrini $s$ adım aralıklı gecikmelere ($x_{t-12}, x_{t-24}, \dots$) de uygulayarak ikincisini ekler.

**Tanım:** SARIMA(p,d,q)(P,D,Q)[s] modeli, gecikme operatörüyle şöyle yazılır:

```math
\underbrace{\Phi(B^s)}_{\substack{\text{mevsimsel}\\\text{AR}}}\, \underbrace{\phi(B)}_{\text{AR}}\, \overbrace{(1 - B)^d}^{d\text{ normal fark}}\, \overbrace{(1 - B^s)^D}^{D\text{ mevsimsel fark}}\, x_t = \underbrace{\theta(B)}_{\text{MA}}\, \underbrace{\Theta(B^s)}_{\substack{\text{mevsimsel}\\\text{MA}}}\, \varepsilon_t
```

**Ne anlatıyor?** Seriye $d$ normal ve $D$ mevsimsel fark uygulandıktan sonra, ardışık dönemler arasındaki ilişki küçük harfli polinomlarla, $s$ dönem arayla tekrarlanan mevsimsel ilişki ise büyük harfli polinomlarla modellenir. Böylece aynı AR/I/MA fikri hem kısa gecikmelere hem de mevsimsel gecikmelere uygulanır.

Burada mevsimsel polinomlar $B$ yerine $B^s$ içerir:

```math
\begin{aligned}
\underbrace{\Phi(B^s)}_{\substack{\text{mevsimsel}\\\text{AR polinomu}}} &= 1 - \underbrace{\Phi_1 B^s - \Phi_2 B^{2s} - \dots - \Phi_P B^{Ps}}_{\substack{P\text{ geçmiş mevsimin}\\\text{katsayıları}}} \\[6pt]
\underbrace{\Theta(B^s)}_{\substack{\text{mevsimsel}\\\text{MA polinomu}}} &= 1 + \underbrace{\Theta_1 B^s + \Theta_2 B^{2s} + \dots + \Theta_Q B^{Qs}}_{\substack{Q\text{ geçmiş mevsimsel}\\\text{şokun katsayıları}}}
\end{aligned}
```

**Ne anlatıyor?** Mevsimsel polinomlar, normal AR ve MA polinomlarıyla aynı yapıdadır; tek fark, gecikmelerin bir adım yerine $s$ adımlık aralıklarla (ör. geçen yılın aynı ayı) atılmasıdır.

> **Simge notu:** $`\Phi_i`$ *(büyük fi)*: mevsimsel AR katsayısı · $`\Theta_j`$ *(büyük teta)*: mevsimsel MA katsayısı · $`s`$: mevsim uzunluğu (aylık veride 12) · $`B^s`$ *(be üzeri s)*: $`s`$ adım geri kaydırma, örneğin $`B^{12} x_t = x_{t-12}`$ (geçen yılın aynı ayı)

Formül uzun görünse de yalnızca 7.2'deki parçaların yan yana yazılmasıdır: Sol tarafta mevsimsel ve mevsimsel olmayan AR polinomları ile iki tür fark, sağ tarafta iki MA polinomu vardır. Parametrelerin anlamı Şekil 7.3'te özetlenmiştir:

- **(p, d, q):** Mevsimsel olmayan kısım (ardışık gözlemler arası ilişki).
- **(P, D, Q):** Mevsimsel kısım. $P$ geçmiş mevsimlerin değerlerini, $D$ mevsimsel fark sayısını, $Q$ geçmiş mevsimlerin hatalarını ifade eder.
- **[s]:** Mevsim uzunluğu. Aylık veride 12, çeyreklik veride 4, saatlik veride günlük döngü için 24.

**Günlük hayattan bir örnek:** Bir doğal gaz faturası değerlendirilirken bu Ocak'ın faturası Aralık'la değil geçen Ocak'la karşılaştırılır; kış zaten her yıl pahalıdır. Bu mevsimsel farktır ($D = 1$, $s = 12$). Bu Ocak tahmin edilirken iki tür sürprize bakılır: geçen ayın beklenmedik sapmasına (küçük harfler, örneğin $q$) ve geçen Ocak'ın beklenmedik sapmasına (büyük harfler, örneğin $Q$). Aşağıdaki havayolu modeli (0,1,1)(0,1,1)[12] tam olarak bu iki sürprizi kullanır.

![SARIMA notasyonu](../images/ch07_sarima_notasyonu.svg)

*Şekil 7.3 — SARIMA notasyonunun parçaları. Büyük harfler, küçük harflerin $`s`$ adım aralıklı (mevsimsel) karşılıklarıdır.*

**Örnek (Havayolu modeli):** Box ve Jenkins'in ünlü kitabında `AirPassengers` için önerilen, bu yüzden literatürde **"airline model"** olarak anılan ARIMA(0,1,1)(0,1,1)[12] modeli şöyle yazılır:

```math
\underbrace{(1 - B)}_{\text{aylık fark}}\, \overbrace{(1 - B^{12})}^{\text{yıllık fark}}\, \underbrace{\log x_t}_{\substack{\text{log yolcu}\\\text{sayısı}}} = \underbrace{(1 + \theta_1 B)}_{\text{aylık MA(1)}}\, \overbrace{(1 + \Theta_1 B^{12})}^{\text{yıllık MA(1)}}\, \varepsilon_t
```

**Ne anlatıyor?** Log serinin hem aylık hem yıllık farkı alınır; kalan seri, geçen ayın ve geçen yılın aynı ayının şoklarıyla açıklanır. Modelde yalnızca iki MA katsayısı bulunur.

Bu denklemi parantezleri çarparak açalım. Gecikme polinomları sıradan cebirdeki gibi çarpılır: Birinci parantezin her terimi ikincinin her terimiyle çarpılır. Tek yeni kural üslerin toplanmasıdır: $B \cdot B^{12} = B^{13}$, çünkü önce 1 adım, sonra 12 adım geri gitmek toplam 13 adım geri gitmek demektir.

*Sağ taraf (şoklar):*

```math
\begin{aligned}
(1 + \theta_1 B)(1 + \Theta_1 B^{12}) &= 1 \cdot 1 + 1 \cdot \Theta_1 B^{12} + \theta_1 B \cdot 1 + \theta_1 B \cdot \Theta_1 B^{12} \\
&= 1 + \theta_1 B + \Theta_1 B^{12} + \theta_1 \Theta_1 B^{13}
\end{aligned}
```

Bu polinomu $\varepsilon_t$'ye uygularken her $B^k$ terimi şoku $k$ ay geri kaydırır ve şu ifade çıkar: $\varepsilon_t + \theta_1 \varepsilon_{t-1} + \Theta_1 \varepsilon_{t-12} + \theta_1 \Theta_1 \varepsilon_{t-13}$. Dördüncü terim modelde ayrıca tanımlanmamıştır; iki parantezin çarpımından kendiliğinden doğar. Sezgisi şöyledir: Aylık düzeltme ($\theta_1$) geçen ayın şokunu, yıllık düzeltme ($\Theta_1$) geçen yılın aynı ayının şokunu kullanır. Geçen yılın aynı ayından **bir ay önceki** şok (13 ay önce) ise hem yıllık hem aylık düzeltmeden geçtiği için iki katsayının çarpımı kadar ağırlık alır.

*Sol taraf (seri):* Kısaca $l_t = \log x_t$ yazalım:

```math
\begin{aligned}
(1 - B)(1 - B^{12})\, l_t &= (1 - B - B^{12} + B^{13})\, l_t \\
&= l_t - l_{t-1} - l_{t-12} + l_{t-13} \\
&= (l_t - l_{t-12}) - (l_{t-1} - l_{t-13})
\end{aligned}
```

Son satır "bu ayın geçen yıla göre (log) değişimi eksi geçen ayın geçen yıla göre değişimi" demektir. Farkı alınmış seriye 7.1.4'teki gibi $w_t$ diyelim (burada iki fark birden alınmıştır); bu, Şekil 7.1c'deki durağanlaştırılmış seridir. Model böylece şu hâli alır:

```math
\underbrace{w_t}_{\substack{\text{durağanlaştırılmış}\\\text{seri}}} = \overbrace{\varepsilon_t}^{\text{bu ayın şoku}} + \underbrace{\theta_1 \varepsilon_{t-1}}_{\text{geçen ayın şoku}} + \overbrace{\Theta_1 \varepsilon_{t-12}}^{\substack{\text{geçen yılın}\\\text{aynı ayının şoku}}} + \underbrace{\theta_1 \Theta_1 \varepsilon_{t-13}}_{\substack{\text{iki parantezin}\\\text{çarpımından doğan terim}}}
```

**Ne anlatıyor?** Durağanlaştırılmış seri; bu ayın, geçen ayın, geçen yılın aynı ayının ve ondan bir ay önceki ayın şoklarının ağırlıklı toplamıdır. Son terimin katsayısı ayrıca kestirilmez, iki katsayının çarpımıdır.

7.6.5'te R'ın bulduğu katsayılarla ($\theta_1 = -0.4018$, $\Theta_1 = -0.5569$) çapraz terimin katsayısı $(-0.4018) \cdot (-0.5569) \approx 0.224$ olur (iki negatif sayının çarpımı pozitiftir). Yani kestirilen model $w_t = \varepsilon_t - 0.402\thinspace  \varepsilon_{t-1} - 0.557\thinspace  \varepsilon_{t-12} + 0.224\thinspace  \varepsilon_{t-13}$'tür; iki katsayıdan dört terimli bir şok yapısı elde edilmiştir.

Bu model 7.6'daki R uygulamasında `auto.arima()` tarafından da seçilecektir.

> **Hemen deneyin 7.3 · Havayolu modeli Python'da** (Python, ~4 dk)
>
> R'ın 7.6.5'te bulacağı katsayıları ($\theta_1 = -0.4018$, $\Theta_1 = -0.5569$) `statsmodels` ile karşılaştıralım:
>
> ```python
> import numpy as np
> import pandas as pd
> from statsmodels.tsa.arima.model import ARIMA
>
> seri = pd.read_csv("data/AirPassengers.csv", index_col="Month",
>                    parse_dates=True)["Passengers"].asfreq("MS")   # "MS": aylık, ay başı
> model = ARIMA(np.log(seri), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
> print(model.params.round(4))
> ```
>
> `seasonal_order=(P, D, Q, s)` mevsimsel kısmı verir; `(0, 1, 1, 12)` = (0,1,1)[12]. `.asfreq("MS")` serinin aylık olduğunu `statsmodels`'a bildirir.
>
> **Ne göreceksiniz:** `ma.L1 -0.4017`, `ma.S.L12 -0.5571`, `sigma2 0.0013`. Katsayılar R'ın sonuçlarıyla (−0.4018, −0.5569) üçüncü ondalığa kadar örtüşür; `ma.S.L12` R'daki `sma1`'in, yani $\Theta_1$'in karşılığıdır.
>
> **Şimdi siz:** Python'un katsayılarıyla 13. gecikmedeki çapraz terimin katsayısını ($\theta_1 \Theta_1$) hesaplayın: `model.params["ma.L1"] * model.params["ma.S.L12"]`. Metindeki değerle aynı mı?
>
> <details><summary>Cevap</summary>
>
> Yaklaşık 0.224: $(-0.4017) \cdot (-0.5571) \approx 0.2238$. Metinde R'ın katsayılarıyla bulunan 0.224 ile aynıdır. Model bu terimi ayrıca kestirmez; iki katsayının çarpımından kendiliğinden doğar.
>
> </details>

---

### 7.4. Model Kurma Süreci: Box-Jenkins Yöntemi

George Box ve Gwilym Jenkins, 1970'te ARIMA modellerinin nasıl kurulacağını sistematik bir döngü olarak tarif ettiler. Bugün de kullanılan bu yöntem, bir hazırlık adımının ardından dört aşamadan oluşur (Şekil 7.4):

![Box-Jenkins döngüsü](../images/ch07_box_jenkins.svg)

*Şekil 7.4 — Box-Jenkins yöntemi. Teşhis aşamasında artıklarda hâlâ yapı varsa model yeniden tanımlanır.*

**0. Hazırlık.** Seri çizilir; trend, mevsimsellik ve varyans değişimi gözle incelenir. Varyans seviyeyle birlikte büyüyorsa log (ya da Box-Cox: log dönüşümünü de içeren güç dönüşümleri ailesi) dönüşümü uygulanır (Bölüm 2.4). Durağanlık ADF ve KPSS testleriyle sınanır. Bu testlerin nasıl çalıştığını 7.5'te ayrıntılı göreceğiz; bu bölüm yöntemin yol haritasıdır.

**1. Tanımlama (identification).** Seriyi durağanlaştıracak fark sayıları ($d$ ve $D$) belirlenir. Ardından durağan serinin ACF ve PACF grafiklerinden (Bölüm 6.7) $p$, $q$, $P$, $Q$ için aday değerler çıkarılır. Kural olarak mevsimsel olmayan terimlere küçük gecikmelerde (1, 2, 3), mevsimsel terimlere $s$'nin katlarında (12, 24, …) bakılır. Mevsimsel gecikmelerde ACF $s$'de tek bir çubuktan sonra kesiliyor ve PACF bu gecikmelerde sönüyorsa mevsimsel MA ($Q = 1$); PACF $s$'den sonra kesilip ACF sönüyorsa mevsimsel AR ($P = 1$) adayı vardır (Bölüm 6.7.5'teki mevsimsel imza tablosu).

**2. Kestirim (estimation).** Aday modellerin katsayıları ($\phi$, $\theta$, $\Phi$, $\Theta$) veriden kestirilir. (Bu derste *kestirim*, katsayıların veriden bulunmasını; *tahmin* ya da *öngörü* (*forecasting*) ise geleceğe ait değerlerin üretilmesini anlatır.) Yazılımlar bunu **en çok olabilirlik** (*maximum likelihood*) yöntemiyle yapar. **Olabilirlik** şu sorunun cevabıdır: "Model ve katsayıları doğru olsaydı, elimizdeki veriyi tam olarak bu hâliyle gözlemleme olasılığı ne kadar olurdu?" Yazılım katsayıları bu olasılığı en büyük yapacak şekilde ayarlar; bir dürbünün odak ayarını görüntü en net olana kadar çevirmek gibi. Bu olasılıklar çok küçük sayılar olduğu için pratikte logaritmaları ($\log L$, *log likelihood*) kullanılır. **Daha büyük $\log L$, veriye daha iyi uyum** demektir.

Adaylar arasında seçim için bilgi kriterleri kullanılır. En yaygını Akaike Bilgi Kriteri'dir:

```math
\mathrm{AIC} = \underbrace{-2 \log L}_{\substack{\text{uyum terimi: uyum}\\\text{iyileştikçe küçülür}}} + \underbrace{2k}_{\substack{\text{ceza: parametre}\\\text{başına 2 puan}}}
```

**Ne anlatıyor?** İlk terim modelin veriye uyumunu ölçer ve uyum iyileştikçe küçülür; ikinci terim her parametre için 2 puanlık ceza ekler. Daha düşük AIC, uyum ile sadelik arasında daha iyi bir denge anlamına gelir.

> **Simge notu:** $`L`$: modelin olabilirlik (likelihood) değeri, yani veriyi ne kadar iyi açıkladığı · $`\log L`$: olabilirliğin doğal logaritması · $`k`$: kestirilen parametre sayısı (katsayılar ve beyaz gürültünün varyansı $`\sigma^2`$)

İlk terim ($-2 \log L$) uyum iyileştikçe küçülür; ikinci terim ($2k$) her ek parametre için 2 puan ceza ekler. **Daha küçük AIC daha iyidir.** Ceza neden gereklidir? Modele parametre eklemek $\log L$'yi hemen her zaman biraz artırır, ama bu artış gerçek bir yapıyı değil, verideki tesadüfi gürültünün ezberlenmesini yansıtabilir. AIC, eklenen parametrenin "kendi maliyetini çıkarıp çıkarmadığını" sorar.

**Günlük hayattan bir örnek:** Bir öğrenci sınav öncesi bütün eski soruların cevaplarını ezberlerse (çok parametre) eski sınavlarda tam puan alır, ama yeni sorularda zorlanır (Bölüm 8.1'de bu duruma aşırı öğrenme diyeceğiz). AIC, her ek parametreye 2 puanlık bir "ezber cezası" keser; aşağıda göreceğimiz BIC bu cezayı daha yüksek tutar (parametre başına $\log n$ puan), bu yüzden daha sade modelleri seçer.

**Sayısal karşılaştırma.** 7.6'daki `log(AirPassengers)` serisi için iki aday modelin R'daki sonuçları şöyledir:

| Model | $`\log L`$ | $`k`$ | $`\mathrm{AIC} = -2 \log L + 2k`$ |
| --- | --- | --- | --- |
| ARIMA(0,1,1)(0,1,1)[12] | 244.70 | 3 ($`\theta_1, \Theta_1, \sigma^2`$) | $`-489.40 + 6 = -483.40`$ |
| ARIMA(1,1,1)(0,1,1)[12] | 244.95 | 4 ($`\phi_1, \theta_1, \Theta_1, \sigma^2`$) | $`-489.90 + 8 = -481.90`$ |

İkinci model fazladan bir AR katsayısı taşır ve $\log L$'yi yalnızca 0.25 artırır. Bu küçük iyileşme, eklenen parametrenin 2 puanlık cezasını karşılamaz. İkinci modelin AIC'si daha büyük olduğu için (−481.90, −483.40'tan büyüktür; negatif sayılarda sıfıra daha yakın olan büyüktür) **sade model tercih edilir**.

> **Hemen deneyin 7.4a · Tablodaki sayıları Python'da üretin** (Python, ~4 dk)
>
> 7.3'teki kutuda kurduğumuz havayolu modelinin log olabilirliğini ve AIC'sini yazdıralım:
>
> ```python
> import numpy as np
> import pandas as pd
> from statsmodels.tsa.arima.model import ARIMA
>
> seri = pd.read_csv("data/AirPassengers.csv", index_col="Month",
>                    parse_dates=True)["Passengers"].asfreq("MS")
> model = ARIMA(np.log(seri), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
> print(round(model.llf, 2), round(model.aic, 2))     # log olabilirlik ve AIC
> ```
>
> `model.llf` (*log-likelihood function*) $\log L$'yi, `model.aic` AIC'yi verir.
>
> **Ne göreceksiniz:** `244.7 -483.39`. Tablonun ilk satırıyla (244.70, −483.40) neredeyse aynıdır; son basamaktaki küçük fark, iki yazılımın sayısal hesap ayrıntılarından gelir.
>
> **Şimdi siz:** `order=(1, 1, 1)` ile tekrar kurun. Log olabilirlik ve AIC kaç olur? Tablonun ikinci satırıyla karşılaştırın.
>
> <details><summary>Cevap</summary>
>
> `ar.L1 0.1968`, `ma.L1 -0.5790`, `ma.S.L12 -0.5642`; log olabilirlik 244.95, AIC −481.89. Tablonun ikinci satırıyla aynıdır: Ek AR katsayısı AIC'yi iyileştirmez, sade model tercih edilir.
>
> </details>

BIC benzer bir kriterdir ama parametre cezası daha ağırdır: $2k$ yerine $k \log n$ ($n$: gözlem sayısı). Farkı alınmış `AirPassengers` serisinde $n = 131$ olduğundan $\log 131 \approx 4.88$'dir ve her parametre 2 yerine yaklaşık 4.9 puan ceza alır; bu yüzden BIC daha sade modelleri tercih eder. AICc ise küçük örneklemler için düzeltilmiş AIC'dir; `auto.arima()` varsayılan olarak AICc'ye bakar.

```math
\mathrm{BIC} = \underbrace{-2 \log L}_{\text{AIC ile aynı uyum terimi}} + \underbrace{k \log n}_{\substack{\text{ceza: parametre}\\\text{başına }\log n\text{ puan}}}
```

**Ne anlatıyor?** Uyum terimi AIC ile aynıdır, ancak parametre başına ceza gözlem sayısıyla büyür. $n$ en az 8 olduğunda $\log n$, 2'yi aştığı için ($\log 8 \approx 2.08$) BIC her parametreye AIC'den daha ağır ceza keser.

**Not —** AIC'nin mutlak değeri anlamsızdır; yalnızca aynı veri üzerindeki (aynı dönüşüm, aynı $d$ ve $D$) modeller karşılaştırılır. Örneğin 7.7.4'teki AIC 536 (log alınmamış 84 aylık eğitim verisi, $d = 0$) ile R'daki −483 (144 aylık log seri, $d = 1$) karşılaştırılamaz.

**3. Teşhis (diagnostic checking).** Modelin açıklayamadığı kısma, yani **artıklara** (residuals) bakılır: $e_t = x_t - \hat{x}_ t$. Kurulan model, verideki hikâyeyi açıklamaya çalışan bir dedektife benzetilebilir. Artıklar, dedektifin açıklayamayıp geride bıraktığı kırıntılardır. Artıklarda bir desen varsa (örneğin her yaz hatalar büyüyorsa) dedektif önemli bir ipucunu kaçırmış demektir. Amacımız artıkların, boş bir radyo kanalındaki cızırtı gibi, tamamen rastgele ve öngörülemez olmasıdır (beyaz gürültü; Bölüm 2.2, 6.7.1).

İyi bir modelin artıklarının üç özelliği olmalıdır:

1. **Ortalaması sıfır olmalı:** Model sistematik olarak ne yukarı ne aşağı yönde hata yapmalı.
2. **Varyansı sabit olmalı:** Hataların büyüklüğü zamanla değişmemeli. Hatalar büyüyorsa tahmin aralıkları güvenilmez olur.
3. **Otokorelasyon içermemeli:** En önemlisi budur. Bir dönemin hatası bir sonrakini tahmin etmeye yardım ediyorsa, model kullanılabilecek bir bilgiyi kaçırmıştır.

Ek olarak artıkların **normal dağılıma yakın** olması, tahmin aralıklarının doğru hesaplanması için istenir. Uygulamadaki denetim 7.6.6'dadır.

Üçüncü özelliğin formel testi **Ljung-Box** testidir. Fikri basittir: Artıkların 1, 2, …, $h$ gecikmedeki otokorelasyonlarını (Bölüm 6.3) hesapla, her birinin **karesini** al (böylece artı ve eksi değerler birbirini götürmez) ve hepsini tek bir sayıda topla:

```math
Q^{\ast} = \underbrace{n(n+2)}_{\substack{\text{kısa seri}\\\text{düzeltmesi}}} \underbrace{\sum_{j=1}^{h} \frac{\overbrace{r_j^2}^{\text{otokorelasyonun karesi}}}{n-j}}_{\substack{\text{ilk }h\text{ gecikme için}\\\text{ağırlıklı toplam}}}
```

**Ne anlatıyor?** Artıkların ilk $h$ gecikmedeki otokorelasyonlarının kareleri, kısa seriler için düzeltilerek tek bir sayıda toplanır. İstatistiğin büyük çıkması, artıklarda modelin yakalayamadığı bir otokorelasyon bulunduğuna işaret eder.

> **Simge notu:** $`\hat{x}_t`$ *(x şapka)*: modelin $`t`$ anı için ürettiği tahmin · $`n`$: gözlem (artık) sayısı · $`r_j`$ *(r j)*: artıkların $`j`$ gecikmedeki örneklem otokorelasyonu · $`\sum_{j=1}^{h}`$ *(büyük sigma, toplam)*: $`j = 1`$'den $`h`$'ye kadar her terimi hesaplayıp topla (gecikme sırası için $`j`$ kullanılır; $`k`$ bu bölümde AIC'de parametre sayısını, ADF'de gecikmeli fark sayısını gösterir) · $`Q^{\ast}`$ *(Q yıldız)*: Ljung-Box test istatistiği

$n(n+2)$ ve $n - j$ çarpanları, kısa serilerde otokorelasyon ölçümündeki küçük sapmaları düzelten ağırlıklardır. Küçük bir örnek: 100 gözlemlik bir seri (bir modelin artıkları değil, ham bir seri; $n = 100$), $h = 2$ gecikme, $r_1 = 0.1$ ve $r_2 = -0.05$ olsun:

```math
Q^{\ast} = 100 \cdot 102 \cdot \left( \frac{0.1^2}{99} + \frac{(-0.05)^2}{98} \right) = 10200 \cdot (0.000101 + 0.0000255) \approx 1.29
```

Artıklar gerçekten beyaz gürültüyse $Q^{\ast}$'ın ortalama değeri, testin **serbestlik derecesi** (`df`) denen sayı kadardır; bu örnekte gecikme sayısına eşit, yani 2'dir (bir modelin artıklarında ise kestirilen katsayı sayısı kadar düşülür; 7.6.6). 1.29 bu yüzden şüphe uyandırmaz. Yazılım p-değerini, $Q^{\ast}$'ı "artıklar saf gürültü olsaydı $Q^{\ast}$ hangi değerleri alırdı?" sorusunun dağılımıyla (**ki-kare dağılımı**) karşılaştırarak hesaplar.

Kısacası ilk $h$ gecikmedeki otokorelasyonlar toplu olarak sıfıra yakınsa $Q^{\ast}$ küçük çıkar. Sıfır hipotezi $H_0$: "artıklar arasında otokorelasyon yoktur" şeklindedir. Bu yüzden burada **yüksek p-değeri (> 0.05) iyi haberdir.** Bu, ADF'deki alışkanlığın tersidir: ADF'de küçük p-değeri aranır, burada ise $H_0$ "her şey yolunda" iddiası olduğu için büyük p-değeri istenir. Artıklarda yapı kalmışsa 1. adıma dönülür.

> **Simge notu:** $`H_0`$ *(H sıfır)*: sıfır hipotezi, testin "varsayılan" iddiası

**4. Öngörü (forecasting).** Teşhisten geçen model geleceği tahmin etmek için kullanılır. Nokta tahminlerinin yanında, belirsizliği gösteren **tahmin aralıkları** (%80, %95) da raporlanmalıdır. %95 tahmin aralığı, gerçek değerin %95 olasılıkla içine düşmesi beklenen alt ve üst sınırdır.

**Not —** R'daki `auto.arima()` ve Python'daki `pmdarima.auto_arima()` fonksiyonları 1. ve 2. adımları otomatikleştirir: Fark sayılarını istatistiksel testlerle ya da mevsimsellik gücü gibi ölçütlerle belirler (7.6.3), sonra farklı (p,q)(P,Q) kombinasyonlarını deneyip en küçük AICc/AIC değerli modeli seçer. **Teşhis adımı ise yine analizi yapan kişiye kalır.** Otomatik seçilen bir model de artık testlerinden kalabilir.

> **Hemen deneyin 7.4b · Ljung-Box'ı elle hesaplayın** (R, ~4 dk)
>
> ```r
> set.seed(5)
> e <- rnorm(100)                                   # 100 "artık" (burada saf gürültü)
> r <- acf(e, lag.max = 2, plot = FALSE)$acf[2:3]   # r_1 ve r_2
> n <- 100
> n * (n + 2) * sum(r^2 / (n - 1:2))                # Q* formülü, h = 2
> Box.test(e, lag = 2, type = "Ljung-Box")          # hazır fonksiyon
> ```
>
> `1:2` iki elemanlı `c(1, 2)` vektörüdür; `r^2 / (n - 1:2)` iki kesri birlikte hesaplar, `sum()` toplar.
>
> **Ne göreceksiniz:** `[1] 3.496648` ve `X-squared = 3.4966, df = 2, p-value = 0.1741`. Elle hesap ile hazır fonksiyon aynıdır ($r_1 = 0.0715$, $r_2 = 0.1689$). p > 0.05: Artıklar beyaz gürültüden ayırt edilemiyor (`Box.test()`'te modelden gelmeyen saf seri için `df` = gecikme sayısıdır).
>
> **Şimdi siz:** Hafızası olan bir seri deneyin: `set.seed(5)`, `ar <- arima.sim(model = list(ar = 0.5), n = 100)`, `Box.test(ar, lag = 2, type = "Ljung-Box")`. Sonuç ne der?
>
> <details><summary>Cevap</summary>
>
> `X-squared = 45.672, df = 2, p-value = 1.209e-10`. p-değeri sıfıra çok yakındır; "otokorelasyon yok" hipotezi reddedilir. Bir modelin artıkları böyle çıksaydı model yapıyı kaçırmış olurdu.
>
> </details>

> **Hemen deneyin 7.4c · Gretl'de iki aday, iki kriter** (Gretl menüleri, ~6 dk)
>
> 1. **File → Open data → Sample file…** ile **Gretl** sekmesinden `nile` dosyasını açın (Nil nehri yıllık akışı, 1871–1970).
> 2. **Model → Univariate time series → ARIMA…** seçin. **Dependent variable**: `nile`; **AR order** 0, **Difference** 1, **MA order** 1; **Include a constant** işaretini kaldırın; **OK**.
> 3. Aynı adımları **AR order** 1 ile tekrarlayın.
>
> Betik karşılığı:
>
> ```gretl
> open nile
> arima 0 1 1 ; nile --nc
> arima 1 1 1 ; nile --nc
> ```
>
> **Ne göreceksiniz:** ARIMA(0,1,1): `theta_1 = -0.732942`, `Akaike criterion 1269.091`, `Schwarz criterion 1274.281`. ARIMA(1,1,1): `phi_1 = 0.254376`, `theta_1 = -0.874137`, `Akaike criterion 1267.255`, `Schwarz criterion 1275.040`. (Gretl BIC'yi *Schwarz criterion* adıyla yazar.) İki modelde de $d = 1$ olduğu için kriterler karşılaştırılabilir.
>
> **Şimdi siz:** Hangi model AIC'ye, hangisi BIC'ye göre seçilir? Neden farklı?
>
> <details><summary>Cevap</summary>
>
> AIC ARIMA(1,1,1)'i (1267.26 < 1269.09), BIC ise ARIMA(0,1,1)'i (1274.28 < 1275.04) seçer. Ek AR katsayısı log olabilirliği yaklaşık 1.9 artırır; bu, AIC'nin 2 puanlık cezasını karşılar ama BIC'nin yaklaşık $\log 99 \approx 4.6$ puanlık cezasını karşılamaz. Kriterler çeliştiğinde teşhis (artık testleri) ve sadelik ilkesi karar verir. R'da `Arima(Nile, order = c(1, 1, 1))` aynı sayıları verir.
>
> </details>

---

### 7.5. Durağanlık Testleri: ADF ve KPSS

ADF testinin mantığı, hipotezleri ve p-değerinin okunuşu Bölüm 3.2.5'te anlatıldı. Burada yalnızca modellemede gereken ayrıntılara bakacağız: ADF regresyonunun tam biçimi, test istatistiğinin dağılımı ve kritik değerler, yazılımlar arasındaki farklar ve hipotezleri ADF'nin tersine kuran KPSS testi.

**ADF'nin çekirdeği.** 3.2.5'te AR(1) denklemini $\nabla x_t = (\phi - 1)\thinspace x_{t-1} + \varepsilon_t$ biçiminde yazmıştık. Parantezdeki sayıya $\gamma$ adını verirsek ADF'nin çekirdek denklemi $\nabla x_t = \gamma\thinspace x_{t-1} + \varepsilon_t$ olur: $\gamma = 0$ ise seri birim köklüdür, $\gamma < 0$ ise ortalamasına geri çekilir. Daha kesin olarak $-2 < \gamma < 0$ (yani $-1 < \phi < 1$) durağanlık demektir; pratikte $\phi$'nin −1'in altına indiği seriler çok nadir olduğundan test yalnızca $\gamma < 0$ yönüne bakar ($\phi$, kök ve $\gamma$ arasındaki ilişki için bkz. 3.2.2–3.2.5).

> **Simge notu:** $`\gamma`$ *(gama)*: $`\phi - 1`$; ADF'nin sınadığı "birim kök katsayısı". Bölüm 3.2.1 ve 6.3.3'teki otokovaryans fonksiyonu $`\gamma(h)`$ ile karıştırılmamalıdır; burada parantez ve $`h`$ yoktur

"$`\phi`$ 1'e eşit mi?" sorusunu "$\gamma$ sıfır mı?" biçiminde sormanın pratik nedeni, regresyon yazılımlarının her katsayı için "bu katsayı sıfır mı?" sorusuna hazır bir ölçü (t istatistiği) üretmesidir.

Şekil 7.5 bu fikri görselleştirir. Alt panellerde her nokta bir dönemi temsil eder: Yatay eksende dünkü seviye, dikey eksende bugünkü değişim vardır. Durağan seride ($\phi = 0.5$) nokta bulutu aşağı eğimlidir: Seviye yüksekken değişim çoğunlukla eksi, düşükken artıdır; seri hep ortalamasına geri çekilir. Rastgele yürüyüşte ($\phi = 1$) bulut yataydır: Dünkü seviye ne olursa olsun bugünkü değişim aynı şekilde dağılır; seriyi geri çeken bir kuvvet yoktur. ADF testi, bulutun içinden geçen doğrunun eğiminin ($\gamma$) sıfırdan anlamlı biçimde küçük olup olmadığını sınar.

![ADF testinin fikri](../images/ch07_adf_fikri.svg)

*Şekil 7.5 — ADF testinin fikri. Aynı rastgele şoklarla üretilmiş iki seri: (a) $`\phi = 0.5`$ olan durağan AR(1), (b) $`\phi = 1`$ olan rastgele yürüyüş. Alt panellerde bugünkü değişim ($`\nabla x_t`$) dünkü seviyeye ($`x_{t-1}`$) karşı çizilmiştir. Uydurulan doğrunun eğimi $`\hat{\gamma}`$, $`\phi - 1`$'in veriden kestirimidir: (a)'da yaklaşık −0.5, (b)'de yaklaşık 0.*

**Tanım 1 (ADF testi — Augmented Dickey-Fuller):** Gerçek serilerde yukarıdaki çekirdek denkleme üç parça eklenir. Serinin farkı; bir sabit, bir trend, serinin bir önceki seviyesi ve gecikmeli farkları üzerine regresyonla açıklanır:

```math
\underbrace{\nabla x_t}_{\text{bugünkü değişim}} = \underbrace{\alpha + \beta t}_{\substack{\text{sabit ve}\\\text{trend}}} + \overbrace{\gamma x_{t-1}}^{\substack{\text{birim kök terimi}\\\text{(sınanan kısım)}}} + \underbrace{\sum_{i=1}^{k} \delta_i \nabla x_{t-i}}_{\substack{\text{geçmiş }k\text{ değişim}\\\text{(genişletilmiş kısım)}}} + \varepsilon_t
```

**Ne anlatıyor?** Serinin bugünkü değişimi; sabit ve trend, bir önceki seviye ve geçmiş $k$ değişim üzerine regresyonla açıklanır. Test, bir önceki seviyenin katsayısının ($\gamma$) sıfırdan anlamlı biçimde küçük olup olmadığını sınar; geçmiş değişimler ise kısa dönem hafızayı denetlemek için denkleme eklenir.

> **Simge notu:** $`\alpha`$ *(alfa)*: sabit terim · $`\beta`$ *(beta)*: doğrusal trend katsayısı · $`t`$: burada zaman sayacı (1, 2, 3, …) · $`\gamma`$ *(gama)*: birim kök katsayısı, $`\phi - 1`$ · $`\delta_i`$ *(delta i)*: $`i`$ dönem önceki değişimin katsayısı · $`\sum_{i=1}^{k}`$: $`i = 1`$'den $`k`$'ya kadar toplam · $`k`$: eklenen gecikmeli fark sayısı

- **$\alpha$ (sabit):** Serinin sıfır yerine örneğin 50 etrafında dalgalanmasına izin verir. Sabit olmasaydı test, ortalaması sıfır olmayan durağan serileri yanlış değerlendirirdi.
- **$\beta t$ (doğrusal trend):** $t$ zaman sayacı, $\beta$ her dönem eklenen sabit artıştır. Bu terim sayesinde alternatif hipotez "seri düz bir trend çizgisi etrafında durağan dalgalanıyor" (**trend-durağan**, Bölüm 3.2) biçimini alır. Yani trend terimli ADF testinde $H_0$'ın reddi "trend çizgisi çıkarılınca seri durağan" demektir; "ortalaması sabit" demek değildir.
- **$\sum \delta_i \nabla x_{t-i}$ (genişletilmiş kısım):** Gerçek seriler AR(1)'den daha uzun bir hafızaya sahip olabilir; bugünkü değişim dünkü ve evvelki günkü değişimlerle de ilişkili olabilir. Bu ilişki denklemde yer almazsa hata terimine sızar, $\varepsilon_t$ artık beyaz gürültü olmaz ve testin p-değerleri yanlış çıkar. Denkleme $k$ adet geçmiş değişim eklemek bu kısa dönem hafızayı "emer" ve $\gamma$'nın temiz ölçülmesini sağlar. Örneğin $k = 2$ ise toplam sembolü $\delta_1 \nabla x_{t-1} + \delta_2 \nabla x_{t-2}$ demektir. Bu terimler olmadan test yalnızca "Dickey-Fuller testi"dir.

**Hipotezler ve tek yönlü test:** 3.2.5'teki hipotezler $\gamma$ ile şöyle yazılır: $H_0$: $\gamma = 0$; $H_1$: $\gamma < 0$ (trend terimi varsa trend etrafında durağan). Alternatif hipotez tek yönlüdür ($\gamma < 0$): $\gamma > 0$ patlayan bir seri demektir ve durağanlık lehine kanıt sayılmaz; R çıktısındaki `alternative hypothesis: stationary` bunu belirtir.

**Test istatistiği ve kritik değerler.** Yazılım regresyonu kurar, $\gamma$'yı veriden kestirir ($\hat{\gamma}$, "gama şapka") ve bu kestirimi kendi **standart hatasına** böler:

```math
\mathrm{DF} = \frac{\overbrace{\hat{\gamma}}^{\text{gama kestirimi}}}{\underbrace{\mathrm{s.e.}(\hat{\gamma})}_{\text{kestirimin standart hatası}}}
```

**Ne anlatıyor?** ADF istatistiği, kestirilen $\gamma$ katsayısının kendi standart hatasına bölünmesiyle elde edilir; istatistik eksi yönde büyüdükçe birim kök hipotezi aleyhine kanıt güçlenir.

Standart hata (s.e.), kestirimin ne kadar belirsiz olduğunu ölçer; oran "kestirim sıfırdan kaç standart hata uzakta?" sorusunun cevabıdır. 7.6.2'deki `AirPassengers` regresyonunda $\hat{\gamma} = -0.7256$ ve standart hatası $0.0991$'dir; $-0.7256 / 0.0991 \approx -7.32$, çıktıdaki `Dickey-Fuller = -7.3186` değeridir (bu ara değerler `adf.test` çıktısında görünmez, yalnızca oranları görünür).

Sıradan bir regresyonda bu oran t-tablosuyla karşılaştırılır; tek yönlü %5 sınırı yaklaşık −1.65'tir. ADF'de bu yapılamaz: $H_0$ doğruyken seri bir rastgele yürüyüştür ve böyle bir seride oran, alışılmış t-dağılımına uymaz; belirgin biçimde negatif değerlere kaymış bir dağılım (**Dickey-Fuller dağılımı**) izler. Bu dağılımın kritik değerleri bilgisayar simülasyonlarıyla tablolaştırılmıştır. Sabit ve trend içeren regresyonda, yaklaşık 140 gözlem için değerler şöyledir:

| Anlamlılık düzeyi | %1 | %5 | %10 |
| --- | --- | --- | --- |
| Kritik değer | −4.03 | −3.44 | −3.14 |

İstatistik kritik değerden daha negatifse $H_0$ reddedilir. −7.32, −4.03'ten de küçük olduğu için birim kök %1 düzeyinde reddedilir. Sıradan t-tablosunu kullansaydık −1.65 gibi çok daha "kolay" bir sınırla, birim köklü serileri gereğinden sık durağan sanırdık. (Trend terimi olmayan, yalnızca sabitli regresyonda %5 kritik değeri yaklaşık −2.88'dir; Python'un `adfuller()` fonksiyonu varsayılan olarak bu biçimi kullanır, 7.7.2.)

**Aynı test, iki yazılım.** ADF testinin varsayılan ayarları yazılıma göre değişir. Aynı seri için farklı sonuçlar görülmesinin başlıca nedeni budur (7.6.2 ve 7.7.2):

| Yazılım | Deterministik terimler (varsayılan) | Gecikme sayısı $k$ (varsayılan) | p-değeri |
| --- | --- | --- | --- |
| R, `tseries::adf.test()` | Sabit + trend; değiştirilemez | $(n-1)^{1/3}$'ün tam sayı kısmı; `k =` ile elle verilir | Kritik değer tablosundan; 0.01–0.99 aralığına kırpılır |
| Python, `statsmodels` `adfuller()` | Yalnızca sabit (`regression="c"`); `"ct"` ile sabit + trend | AIC ile seçilir (`autolag="AIC"`); `autolag=None` ile `maxlag` aynen kullanılır | MacKinnon yaklaşımı; kırpılmaz |

Örneğin ham `AirPassengers` serisinde sabit + trend ve 5 gecikmeyle istatistik −7.3186, sabit + trend ve 13 gecikmeyle −2.1008, yalnızca sabit ve 13 gecikmeyle 0.8154'tür. Ayarlar eşitlendiğinde iki yazılım aynı istatistiği verir (7.7.2'deki kutu).

**Tanım 2 (KPSS testi — Kwiatkowski-Phillips-Schmidt-Shin):** KPSS bakış açısını tersine çevirir: "Seri durağandır" varsayımıyla başlar ve buna karşı kanıt arar.

- $H_0$: **Seri durağandır** (seviye etrafında ya da trend etrafında).
- $H_1$: Seri birim kök içerir.
- Küçük p-değeri (< 0.05) → $H_0$ reddedilir → seri durağan **değildir**.

Testin fikri şöyledir: Seriden ortalamasını (trend sürümünde trend çizgisini) çıkarırız ve geriye kalan sapmaları baştan itibaren **art arda toplarız** (kümülatif toplam, 7.1.4). Seri gerçekten durağansa sapmalar bir artı bir eksi gelir, birbirini götürür ve kümülatif toplam sıfır civarında kalır. Seviye bir yöne kayıyorsa sapmalar uzun süre aynı işaretli olur ve toplam sıfırdan uzaklaşır. Örneğin sapmalar $+2, -1, -2, +1$ ise kümülatif toplamlar $2, 1, -1, 0$'dır. Sapmalar $-3, -2, -1, +1, +2, +3$ ise (önce ortalamanın altında, sonra üstünde, yani kayan bir seviye) kümülatif toplamlar $-3, -5, -6, -5, -3, 0$ olur ve sıfırdan çok daha fazla uzaklaşır. KPSS istatistiği, bu kümülatif toplamların karelerinin büyüklüğünü serinin kendi oynaklığına göre ölçekleyerek tek sayıda özetler. **Büyük istatistik → $H_0$ (durağanlık) reddedilir.** Seviye durağanlığı için kritik değerler %10, %5 ve %1 düzeyinde sırasıyla 0.347, 0.463 ve 0.739'dur.

İki testin sonuçları birlikte yorumlanır. Tablo okunurken önce her testin sonucu sade bir cümleye çevrilir: ADF'de $H_0$'ın reddi "durağan", KPSS'te $H_0$'ın reddi "durağan değil" demektir.

| ADF ($`H_0`$: birim kök) | KPSS ($`H_0`$: durağan) | İki test ne diyor? | Ne yapmalı? |
| --- | --- | --- | --- |
| p < 0.05: $`H_0`$ reddedildi → "durağan" | p ≥ 0.05: $`H_0`$ reddedilmedi → "durağan" | İkisi de **durağan** diyor | **Ek** fark almaya gerek yok (7.6.3'teki farkı alınmış seri) |
| p ≥ 0.05: $`H_0`$ reddedilmedi → "durağan değil" | p < 0.05: $`H_0`$ reddedildi → "durağan değil" | İkisi de **durağan değil** diyor | Fark alınır (gerekirse önce log), testler tekrarlanır |
| p < 0.05 → "durağan" | p < 0.05 → "durağan değil" | **Çelişki:** ADF birim kökü, KPSS de durağanlığı reddediyor | Çoğunlukla trend-durağanlık, güçlü mevsimsellik ya da ADF'ye yetersiz gecikme verilmesi (7.6.2'deki ham `AirPassengers`). Grafiğe bakılır, fark alınıp yeniden test edilir |
| p ≥ 0.05 → "durağan değil" | p ≥ 0.05 → "durağan" | **Kararsızlık:** İki test de kendi $`H_0`$'ını reddedecek kanıt bulamadı | Genellikle seri kısadır ya da $`\phi`$ 1'e çok yakındır; testler ayırt edemiyor. Grafiğe ve ACF'ye bakılır |

**Not —** 3.2.5'teki uyarı burada da geçerlidir: "Reddedilemedi", "kanıtlandı" demek değildir. Bu yüzden tek bir teste değil, grafik + ACF + iki testin birlikte verdiği tabloya güvenilmelidir.

> **Hemen deneyin 7.5a · Üç seri, iki test** (R, ~5 dk)
>
> ```r
> library(tseries)
> adf.test(Nile)$p.value;   kpss.test(Nile)$p.value    # Nil yıllık akışı
> adf.test(lynx)$p.value;   kpss.test(lynx)$p.value    # vaşak sayısı (10 yıllık döngü)
> adf.test(co2)$p.value;    kpss.test(co2)$p.value     # Mauna Loa aylık CO2
> ```
>
> `$p.value` test sonucundan yalnızca p-değerini çeker; `;` aynı satırda iki komut çalıştırır.
>
> **Ne göreceksiniz:**
>
> | Seri | ADF p | KPSS p | Karar (7.5'teki tablo) |
> | --- | --- | --- | --- |
> | `Nile` | 0.064 | 0.01 | İkisi de "durağan değil" |
> | `lynx` | 0.01 | 0.1 | İkisi de "durağan" |
> | `co2` | 0.227 | 0.01 | İkisi de "durağan değil" |
>
> 0.01 ve 0.1 değerlerinin yanında R, Bölüm 7.6.2'deki gibi "p-value smaller/greater than printed p-value" uyarıları verir. `lynx` güçlü bir döngü içermesine rağmen durağandır: Döngü sabit bir ortalamanın çevresinde tekrarlar. `Nile`'da ise 1898 civarında akış kalıcı olarak düşmüştür (seviye kayması).
>
> **Şimdi siz:** `diff(Nile)` için iki testi tekrarlayın. Karar değişir mi?
>
> <details><summary>Cevap</summary>
>
> ADF p = 0.01 (birim kök reddedilir), KPSS p = 0.1 (durağanlık reddedilmez): Birinci farkı alınmış seri durağandır. Yani `Nile` için $d = 1$ makul bir seçimdir; 7.4c'deki modeller de bu yüzden $d = 1$ ile kuruldu.
>
> </details>

> **Hemen deneyin 7.5b · Güneş lekeleri: Gretl'de ADF ve KPSS** (Gretl menüleri, ~5 dk)
>
> Gretl'in örnek dosyalarındaki `sunspots`, 1700–2021 arasındaki yıllık ortalama güneş lekesi sayısıdır (322 gözlem; yaklaşık 11 yıllık bir döngüsü vardır).
>
> 1. **File → Open data → Sample file…** ile **Gretl** sekmesinden `sunspots` dosyasını açın ve `sunspots` değişkenini seçin.
> 2. **Variable → Unit root tests → Augmented Dickey-Fuller test** seçin. Gecikme sayısını (*lag order*) 4 yapın, gecikmeyi otomatik küçülten "test down from maximum lag order" seçeneği işaretliyse kaldırın, "with constant" ve "with constant and trend" kutularını işaretleyip **OK** deyin.
> 3. **Variable → Unit root tests → KPSS test** seçin, gecikme sayısını 4 yapıp **OK** deyin.
>
> Betik karşılığı (menü etiketleri sürüme göre biraz farklı görünebilir; betik sonucu kesindir):
>
> ```gretl
> open sunspots
> adf 4 sunspots --c --ct
> kpss 4 sunspots
> ```
>
> **Ne göreceksiniz:** ADF'de sabitli sürüm için `tau_c(1) = -8.5338`, `asymptotic p-value 1.281e-14`; sabit ve trendli sürüm için `tau_ct(1) = -8.68544`, `asymptotic p-value 8.497e-15`. KPSS'te `Test statistic = 0.186674`, kritik değerler `0.348 0.462 0.741` (Gretl örneklem büyüklüğüne göre düzeltilmiş değerler verdiği için 7.5'teki 0.347, 0.463, 0.739'dan biraz farklıdır) ve `P-value > .10`. ADF birim kökü açıkça reddeder, KPSS durağanlığı reddetmez: İki test de "durağan" diyor (7.5'teki tablonun ilk satırı).
>
> **Şimdi siz:** Bu sonuç 7.5a'daki hangi R serisine benziyor? Belirgin bir döngüsü olan seri neden durağan sayılabilir?
>
> <details><summary>Cevap</summary>
>
> `lynx`'e. İki seride de döngü, zaman içinde değişmeyen bir ortalamanın çevresinde tekrarlar; seviye kalıcı olarak kaymaz ve şoklar sönümlenir. Durağanlık "dalgalanma yok" demek değil, ortalama ve varyansın zamanla değişmemesi demektir. (`Nile`'da ise seviye 1898 civarında kalıcı biçimde düşmüştü.)
>
> </details>

<details>
<summary><b>Kendinizi test edin (cevaplar için tıklayın)</b></summary>

1. **Bir AR(1) modelinde $`\phi = 0.8`$ ise ADF denklemindeki $`\gamma`$ kaçtır? Bu seri durağan mıdır?**
   → $`\gamma = 0.8 - 1 = -0.2`$. $`-2 < \gamma < 0`$ (yani $`-1 < \phi < 1`$) olduğu için seri durağandır. Dün 100 olan seri, şok yoksa bugün $`100 - 0.2 \cdot 100 = 80`$ olur.

2. **Sabit ve trend içeren bir ADF regresyonunda, yaklaşık 140 gözlemle `Dickey-Fuller = -2.10` bulundu. Birim kök %5 düzeyinde reddedilir mi?**
   → Hayır. −2.10, %5 kritik değeri olan −3.44'ten daha negatif değildir. (Bu, 7.6.2'de `AirPassengers` için 13 gecikmeyle bulunan sonuçtur; p ≈ 0.53.)

3. **Karakteristik denklemi $`(1 - z)(1 - 0.4z) = 0`$ olan bir AR(2) süreci için ne söylenebilir?**
   → Kökler $`z = 1`$ ve $`z = 1/0.4 = 2.5`$'tir. Bir birim kök vardır; seri durağan değildir. Model $`(1 - 0.4B)(1 - B)\thinspace x_t = \varepsilon_t`$ biçiminde yazılabilir: Bir kez fark alınınca geriye $`\phi = 0.4`$ olan durağan bir AR(1) kalır, yani model ARIMA(1,1,0)'dır.

4. **Bir modelin artıkları için Ljung-Box p-değeri 0.233 çıktı. Bu iyi mi kötü mü?**
   → İyi. $`H_0`$ "artıklarda otokorelasyon yok" iddiasıdır ve reddedilmemiştir; artıklar beyaz gürültüden ayırt edilemiyor.

</details>

---

### 7.6. R Uygulaması: `AirPassengers` ile SARIMA

> **Uygulama dosyası:** [`Codes/R/ch07_sarima_airpassengers.R`](../Codes/R/ch07_sarima_airpassengers.R)
>
> Bu bölümdeki R kodlarının tamamı bu dosyada. Dosya RStudio'da açılıp satır satır ya da depo kök dizininde `Rscript Codes/R/ch07_sarima_airpassengers.R` komutuyla çalıştırılabilir.


Şimdi Box-Jenkins adımlarını R üzerinde `AirPassengers` veri setiyle uygulayalım. Bu veri seti belirgin bir trend, mevsimsellik ve zamanla artan varyans içerdiği için öğretici bir örnektir. Kod bloklarındaki `#` (yorum) ve `#>` (çıktı) satırları Bölüm 2.5'teki gibi okunur.

#### 7.6.1. Veriyi Görselleştirme

Bir zaman serisi analizine başlarken ilk adım veriyi çizmektir. Grafik bize trend olup olmadığını, düzenli tekrar eden dalgalanmalar (mevsimsellik) bulunup bulunmadığını ve verinin değişkenliğinin zamanla değişip değişmediğini gösterir.

```r
# Gerekli paketler
# install.packages(c("forecast", "tseries"))
library(forecast)
library(tseries)

# Veriyi yükle ve çiz
data(AirPassengers)
plot(AirPassengers, main = "AirPassengers Verisi: Trend ve Artan Varyans",
     ylab = "Yolcu Sayısı", xlab = "Yıl", col = "darkblue")
```

Bu blok, bölüm boyunca kullanacağımız iki paketi yükler, `AirPassengers` verisini çağırır ve seriyi çizer; Box-Jenkins döngüsünün 0. adımı (hazırlık, 7.4) budur. Kodu satır satır okuyalım:

1. `# Gerekli paketler` ve `# install.packages(c("forecast", "tseries"))`: Yorum satırlarıdır; paketler kurulu değilse ikincisinin başındaki `#` silinip satır bir kez çalıştırılır (Bölüm 3.2).
2. `library(forecast)`: Kurulu `forecast` paketini bu oturuma yükler. Bölümde kullanacağımız `auto.arima()`, `forecast()`, `checkresiduals()`, `ndiffs()` ve `nsdiffs()` bu paketten gelir.
3. `library(tseries)`: `adf.test()` ve `kpss.test()` durağanlık testlerini sağlayan `tseries` paketini yükler.
4. `data(AirPassengers)`: R ile birlikte gelen `AirPassengers` veri setini çalışma alanına çağırır. Bu, Ocak 1949'dan Aralık 1960'a kadar 144 aylık uluslararası havayolu yolcu sayısını (bin kişi) tutan, frekansı 12 olan hazır bir `ts` nesnesidir (Bölüm 5.2.3); ayrıca dönüştürmeye gerek yoktur.
5. `plot(AirPassengers, main = ..., ylab = ..., xlab = ..., col = "darkblue")`: `ts` nesnesini zaman eksenli bir çizgi grafiği olarak çizer (Bölüm 6.1.1); `main` başlığı, `ylab` ve `xlab` eksen adlarını, `col` çizgi rengini (`"darkblue"`: koyu mavi) belirler.

Bu blok ekrana sayı yazmaz; çıktısı bir grafiktir (Şekil 7.1a). Grafikte üç şey hemen göze çarpar:

- Yolcu sayısı yıllar içinde sürekli artıyor: **trend** var.
- Her yıl yaz aylarında tepe yapan bir desen tekrarlanıyor: **mevsimsellik** var.
- Dalgaların boyu zamanla büyüyor: **varyans artıyor** (çarpımsal yapı, Bölüm 2.4).

Bu üç özellik de serinin ortalamasının ve varyansının zamanla değiştiğini, yani durağan olmadığını gösterir.

#### 7.6.2. Durağanlık Testleri

Grafikten edindiğimiz "seri durağan değil" izlenimini şimdi 7.5'teki iki testle sınayalım.

```r
adf.test(AirPassengers)
#>  Augmented Dickey-Fuller Test
#> data:  AirPassengers
#> Dickey-Fuller = -7.3186, Lag order = 5, p-value = 0.01
#> alternative hypothesis: stationary
#> Warning: p-value smaller than printed p-value

adf.test(AirPassengers, k = 13)   # 12 aylık hafızayı kapsayan gecikme sayısıyla
#> Dickey-Fuller = -2.1008, Lag order = 13, p-value = 0.5345

kpss.test(AirPassengers)
#>  KPSS Test for Level Stationarity
#> data:  AirPassengers
#> KPSS Level = 2.7395, Truncation lag parameter = 4, p-value = 0.01
#> Warning: p-value smaller than printed p-value
```

Bu blok aynı seriye iki ADF testi ve bir KPSS testi uygular. Amaç hem durağanlık kararını sayılarla desteklemek hem de bir testin ayarlarının sonucu nasıl değiştirebildiğini görmektir. Kodu satır satır okuyalım:

1. `adf.test(AirPassengers)`: `tseries` paketindeki ADF testini varsayılan ayarlarla (sabit + trend terimli regresyon, tek yönlü "durağan" alternatifi; 7.5) çalıştırır.
2. `adf.test(AirPassengers, k = 13)`: Aynı test, bu kez `k` argümanı elle verilerek çalıştırılır. `k = 13`, regresyona 13 gecikmeli fark ($`\nabla x_{t-1}, \dots, \nabla x_{t-13}`$) ekler. 13, 12 aylık mevsim uzunluğunu bir adım aşar. Bu testin çıktısı kısaltılmıştır; yalnızca değişen satır gösterilmiştir.
3. `kpss.test(AirPassengers)`: KPSS testini çalıştırır. Varsayılan `null = "Level"` sıfır hipotezini "seri sabit bir seviye etrafında durağandır" olarak kurar (`null = "Trend"` yazılsaydı "trend çizgisi etrafında durağandır" sınanırdı).

Çıktıyı satır satır okuyalım:

- `Dickey-Fuller = -7.3186`: 7.5'teki $\hat{\gamma} / \mathrm{s.e.}(\hat{\gamma})$ oranıdır ve kritik değerlerle (%1 için −4.03) karşılaştırılır.
- `Lag order = 5`: Denkleme eklenen gecikmeli fark sayısı $k$. $n = 144$ için $(144 - 1)$'in küp kökü yaklaşık 5.23'tür ($5.23^3 \approx 143$); tam sayı kısmı 5'tir.
- `p-value = 0.01` ve `Warning: p-value smaller than printed p-value`: Gerçek p-değeri 0.01'den de küçüktür (uyarı hata değildir; Bölüm 3.2 ve 7.5).
- İkinci testteki `Dickey-Fuller = -2.1008, Lag order = 13, p-value = 0.5345` satırı aynı biçimde okunur; `Lag order` artık elle verdiğimiz 13'tür. p-değeri 0.01 ile 0.99 arasında kaldığı için uyarı yoktur.
- `KPSS Level = 2.7395`: KPSS istatistiği. %1 kritik değeri 0.739 olduğundan p-değeri tablonun dışına, 0.01'in altına düşer; bu yüzden KPSS de `p-value = 0.01` yazıp aynı uyarıyı verir.
- `Truncation lag parameter = 4`: KPSS'in serinin oynaklığını hesaplarken hesaba kattığı otokorelasyon gecikmesi sayısı ($4 \cdot (n/100)^{1/4}$'ün tam sayı kısmı; $n = 144$ için 4).

**Çıktının yorumu:** İlk bakışta şaşırtıcı bir sonuç: ADF testi p = 0.01 ile birim kök hipotezini **reddediyor**, KPSS testi ise p = 0.01 ile durağanlık hipotezini **reddediyor**. Yani iki test çelişiyor (7.5'teki tablonun üçüncü satırı).

Bunun nedeni `tseries::adf.test()` fonksiyonunun iki varsayılan ayarının birleşimidir. Birincisi, regresyona bir **doğrusal trend terimi** ($\beta t$) ekler; test "seri düz bir trend çizgisi etrafında durağan mı?" sorusunu sorar. İkincisi, yalnızca $k = 5$ gecikmeli fark kullanır. Oysa `AirPassengers`'ın aylık değişimleri 12 ay önceki değişimlerle güçlü biçimde ilişkilidir (yıllık desen). Beş gecikme bu hafızayı emmeye yetmez ve geride kalan mevsimsel yapı testi yanıltır. İkinci satırda aynı test `k = 13` ile, yani bir yılı aşan gecikmeyle tekrarlandığında istatistik −2.10'a, p-değeri 0.53'e çıkar ve birim kök artık reddedilmez. Ortalama sabit değildir ve mevsimsel desen ile artan varyans da hâlâ oradadır. Seviye durağanlığını sınayan KPSS ve grafik bu yüzden daha güvenilir bir tablo çiziyor: **seri durağan değildir.** Bu örnek, tek bir teste ve onun varsayılan ayarlarına körü körüne güvenmemek gerektiğini çok iyi gösterir.

#### 7.6.3. Seriyi Durağanlaştırma

Durağanlığı sağlamak için iki işlem yapılır:

1. **Log dönüşümü:** Artan varyansı dengeler (çarpımsal yapıyı toplamsala çevirir).
2. **Fark alma:** Mevsimsel fark (lag = 12) yıllık deseni, normal fark trendi siler.

```r
# Önce log dönüşümü, sonra mevsimsel (lag = 12) ve normal (lag = 1) fark
AP_stationary <- diff(diff(log(AirPassengers), lag = 12))

plot(AP_stationary, main = "Dönüştürülmüş AirPassengers Serisi",
     ylab = "Fark Değerleri", col = "darkblue")
abline(h = 0, lty = 2)

# Testleri tekrarlayalım
adf.test(AP_stationary)
#> Dickey-Fuller = -5.1993, Lag order = 5, p-value = 0.01
#> alternative hypothesis: stationary
#> Warning: p-value smaller than printed p-value

kpss.test(AP_stationary)
#> KPSS Level = 0.084365, Truncation lag parameter = 4, p-value = 0.1
#> Warning: p-value greater than printed p-value
```

Bu blok, 7.1.4'te anlatılan log dönüşümünü ve iki fark alma işlemini tek satırda uygular, sonucu çizer ve 7.6.2'deki testleri dönüştürülmüş seri üzerinde tekrarlar. Amaç, modele verilecek serinin artık durağan olduğunu göstermektir. Kodu satır satır okuyalım:

1. `AP_stationary <- diff(diff(log(AirPassengers), lag = 12))`: Sonuç `<-` ile (atama, 2.5) `AP_stationary` adıyla saklanır. Sağ taraf iç içe üç fonksiyondan oluşur ve içten dışa okunur:
   - `log(AirPassengers)`: Her ayın değerinin doğal logaritmasını alır; sonuç yine 144 aylık bir `ts` nesnesidir.
   - `diff(..., lag = 12)`: `diff()` bir seriden farklar üretir; `lag` argümanı kaç adım önceki değerin çıkarılacağını söyler. `lag = 12`, her aydan 12 ay önceki değeri çıkarır (mevsimsel fark, $`\nabla_{12}`$), çünkü veri aylık ve mevsim 12 aydır.
   - En dıştaki `diff(...)`: `lag` yazılmadığı için varsayılan `lag = 1` ile her değerden bir önceki ayınkini çıkarır (normal fark, $`\nabla`$).

   Mevsimsel fark ilk 12 ayı, normal fark bir ayı daha kaybettirir; 144 aylık seri Şubat 1950'den başlayan 131 aylık bir `ts` nesnesine iner. Sonuç, 7.3'teki $`w_t`$ serisidir.
2. `plot(AP_stationary, main = ..., ylab = ..., col = "darkblue")`: Dönüştürülmüş seriyi çizer; argümanlar 7.6.1'deki `plot()` çağrısıyla aynı anlamdadır.
3. `abline(h = 0, lty = 2)`: Açık olan grafiğin üstüne düz bir çizgi ekler. `h = 0` çizginin 0 yüksekliğinde yatay (*horizontal*) olacağını, `lty = 2` (*line type*, çizgi tipi) kesikli çizileceğini söyler (1 düz, 2 kesikli, 3 noktalı çizgidir). `abline()` yeni grafik açmaz, son çizilen grafiğe ekleme yapar; bu yüzden `plot()`'tan sonra gelir. Durağan serinin bu sıfır çizgisinin çevresinde dalgalanması beklenir.
4. `adf.test(AP_stationary)` ve `kpss.test(AP_stationary)`: 7.6.2'deki iki testi varsayılan ayarlarıyla bu kez dönüştürülmüş seriye uygular. `# Testleri tekrarlayalım` satırı yorumdur.

Serinin ilk değerini (Şubat 1950) elle hesaplayalım: Ocak ve Şubat 1949'da 112 ve 118, Ocak ve Şubat 1950'de 115 ve 126 bin yolcu vardır.

$$
w = [\log(126) - \log(118)] - [\log(115) - \log(112)] = 0.0656 - 0.0264 = 0.0392
$$

Yani Şubat'ın yıllık büyümesi (yaklaşık %6.6), Ocak'ın yıllık büyümesinden (yaklaşık %2.6) yaklaşık 4 puan fazladır. R'da `AP_stationary[1]` de aynı 0.0392 değerini verir.

Çıktıyı okuyalım:

- `Dickey-Fuller = -5.1993, Lag order = 5, p-value = 0.01`: İstatistik −5.20, %1 kritik değeri olan yaklaşık −4.03'ten daha negatiftir; birim kök reddedilir. Gecikme sayısı yine 5'tir, çünkü $(131 - 1)$'in küp kökü yaklaşık 5.07'dir. Uyarı satırı 7.6.2'dekiyle aynı anlamdadır: Gerçek p-değeri 0.01'den de küçüktür.
- `KPSS Level = 0.084365, Truncation lag parameter = 4, p-value = 0.1` ve `Warning: p-value greater than printed p-value`: İstatistik %10 kritik değeri olan 0.347'nin altındadır, yani tablonun en "zararsız" ucunun da ötesindedir. R p-değerini tablonun üst sınırı olan 0.1'de keser ve gerçek değerin bundan **büyük** olduğunu uyarır. Durağanlık reddedilmez.

Artık iki test aynı şeyi söylüyor: ADF birim kökü reddediyor, KPSS durağanlığı reddetmiyor (7.5'teki tablonun ilk satırı). Seri durağandır ve grafikte, Şekil 7.1c'deki gibi sıfır çevresinde dalgalanır.

`forecast` paketi gereken fark sayılarını doğrudan da önerebilir:

```r
nsdiffs(log(AirPassengers))                  # gereken mevsimsel fark sayısı (D)
#> [1] 1
ndiffs(diff(log(AirPassengers), lag = 12))   # mevsimsel farktan sonra gereken normal fark sayısı (d)
#> [1] 1
```

Bu iki satır, fark sayılarına testlere tek tek bakarak karar vermek yerine `forecast` paketinin hazır karar fonksiyonlarına sorar. Kodu satır satır okuyalım:

1. `nsdiffs(log(AirPassengers))`: Log seriye kaç kez **mevsimsel** fark ($D$) alınması gerektiğini önerir. Varsayılan `test = "seas"` yönteminde seri trend, mevsim ve kalan bileşenlerine ayrılır (Bölüm 6.2.3'teki `decompose()` fikrine benzer) ve mevsimsel desenin gücü 0 ile 1 arasında bir sayıyla ölçülür: 0 "mevsimsellik yok", 1 "kalan dalgalanmanın neredeyse tamamı mevsimsel" demektir. Güç 0.64 eşiğini aşarsa bir mevsimsel fark önerilir; `log(AirPassengers)` için güç yaklaşık 0.96'dır. Mevsim uzunluğunu (`m`) serinin frekansından (12) kendisi okur ve varsayılan `max.D = 1` nedeniyle en fazla 1 önerir. Satır sonundaki `#` sonrası yorumdur.
2. `ndiffs(diff(log(AirPassengers), lag = 12))`: Mevsimsel farkı alınmış seriye kaç kez daha **normal** fark ($d$) gerektiğini önerir. İçteki `diff(log(AirPassengers), lag = 12)`, 7.6.3'teki mevsimsel farktır. `ndiffs()` varsayılan olarak KPSS testini (`test = "kpss"`, `alpha = 0.05`) tekrar tekrar uygular: Test durağanlığı reddederse bir fark alır ve yeniden dener; durağanlık reddedilmeyene kadar aldığı fark sayısını döndürür (en fazla `max.d = 2`). Sıra önemlidir: Önce mevsimsel fark kararı verilir, `ndiffs()` mevsimsel farkı alınmış seriye uygulanır.

İki çıktı da `[1] 1`'dir: Önerilen fark sayısı 1'dir. Böylece $D = 1$ ve $d = 1$ kararını hem testlerle hem de bu fonksiyonlarla doğrulamış olduk.

> **Hemen deneyin 7.6.3 · Kaç fark gerekir?** (R, ~3 dk)
>
> ```r
> library(forecast)
> nsdiffs(nottem)                      # mevsimsel fark sayısı D
> ndiffs(diff(nottem, lag = 12))       # mevsimsel farktan sonra normal fark sayısı d
> ```
>
> **Ne göreceksiniz:** Sırasıyla `[1] 1` ve `[1] 0`. Nottingham sıcaklıkları (`nottem`) için $D = 1$, $d = 0$ (`AirPassengers`'ta $d = 1$ idi): Yıllık desen silinince geriye trendsiz, durağan bir seri kalır (Bölüm 6.2.3'te trendin neredeyse düz olduğunu görmüştük).
>
> **Şimdi siz:** Aynı iki satırı `co2` için çalıştırın. Sonucu tahmin edin: CO2 hem mevsimsel hem de sürekli yükselen bir seridir.
>
> <details><summary>Cevap</summary>
>
> `nsdiffs(co2)` 1, `ndiffs(diff(co2, lag = 12))` 1 verir: $D = 1$, $d = 1$. Mevsimsel fark yıllık deseni siler, ama yükselişin hızı da değiştiği için bir normal fark daha gerekir.
>
> </details>

#### 7.6.4. Model Belirleme (ACF ve PACF)

Durağan serinin ACF ve PACF grafiklerini inceleyerek AR ve MA terimleri için ipuçları ararız (Bölüm 6.7).

```r
par(mfrow = c(1, 2))  # grafikleri yan yana göster
acf(AP_stationary, lag.max = 36, main = "ACF")
pacf(AP_stationary, lag.max = 36, main = "PACF")
par(mfrow = c(1, 1))
```

Bu blok, durağanlaştırılmış serinin ACF ve PACF grafiklerini yan yana çizer. Box-Jenkins'in 1. adımında (tanımlama, 7.4) $p$, $q$, $P$, $Q$ için aday değerleri bu grafiklerden okuruz. Kodu satır satır okuyalım:

1. `par(mfrow = c(1, 2))`: `par()` (*parameters*) R'ın temel grafik ayarlarını değiştirir. `mfrow` argümanı grafik penceresini satır ve sütun bölmelerine ayırır; `c(1, 2)` ile kurulan iki elemanlı vektör "1 satır, 2 sütun" demektir. Bundan sonra çizilen grafikler bölmeleri soldan sağa doldurur. Satır sonundaki `#` sonrası yorumdur.
2. `acf(AP_stationary, lag.max = 36, main = "ACF")`: `acf()` serinin otokorelasyonlarını hesaplar ve çubuk grafik olarak çizer (Bölüm 6.3–6.4). `lag.max` kaç gecikmeye kadar bakılacağını belirler: 36 ay üç yıl demektir; böylece 12, 24 ve 36'daki mevsimsel gecikmeler de görünür. Yazılmasaydı varsayılan değer $10 \log_{10} 131 \approx 21$ gecikme olurdu ve 24 ile 36 grafiğe girmezdi. `main` grafik başlığıdır.
3. `pacf(AP_stationary, lag.max = 36, main = "PACF")`: Aynı ayarlarla kısmi otokorelasyonları çizer (Bölüm 6.5) ve ikinci bölmeye yerleşir.
4. `par(mfrow = c(1, 1))`: Pencereyi yeniden tek grafik düzenine döndürür; bu satır olmasaydı sonraki grafikler de yarım genişlikte çizilirdi.

Bu blok ekrana sayı yazmaz; çıktısı yan yana iki grafiktir. Okurken iki ayrıntı önemlidir. Birincisi, R `ts` nesnelerinde yatay ekseni "yıl" cinsinden gösterir; 1.0 = 12 ay, 2.0 = 24 ay gecikme demektir. ACF'nin en soldaki 0 gecikmeli çubuğu serinin kendisiyle korelasyonudur, her zaman 1'dir ve yorumlanmaz; PACF ise 1. gecikmeden başlar. İkincisi, mavi kesikli çizgiler güven bandıdır: $\pm 1.96/\sqrt{131} \approx \pm 0.17$. Bu grafiklerde iki belirgin iz görülür:

- **ACF'de gecikme 1'de** belirgin bir negatif çubuk (yaklaşık −0.34) vardır; 3. gecikmede bandı biraz aşan bir çubuk (yaklaşık −0.20) dışında sonrası büyük ölçüde kesilir; 9, 23 ve 32. gecikmelerdeki sınırı az aşan çubuklar (yaklaşık 0.18, 0.22 ve 0.20) 36 çubukta tesadüfen beklenebilecek düzeydedir (Bölüm 6.3.5) → mevsimsel olmayan kısımda **MA(1)**, yani $q = 1$ adayı.
- **ACF'de gecikme 12'de** (eksende 1.0) belirgin bir negatif çubuk (yaklaşık −0.39) vardır ve 24'te tekrarlamaz → mevsimsel kısımda **MA(1)**, yani $Q = 1$ adayı.
- PACF'de de 1 ve 12'de anlamlı negatif çubuklar (yaklaşık −0.34) vardır, ancak PACF ACF kadar keskin kesilmez: 3. ve 13. gecikmelerde de küçük negatif çubuklar sürer. 9. gecikmedeki pozitif çubuk (yaklaşık 0.23) bandı aşsa da ACF'deki 9. gecikme gibi tek başına kalır ve tesadüf düzeyindedir. "ACF kesiliyor, PACF kesilmiyor" deseni MA yapısıyla tutarlıdır (Bölüm 6.7.2 ve 6.7.5).

Bu okuma bizi ARIMA(0,1,1)(0,1,1)[12] adayına götürür. ACF/PACF okumak deneyim ister; bu yüzden elle bulduğumuz adayı otomatik aramayla karşılaştıracağız.

#### 7.6.5. Model Kurma: `auto.arima()` ve Seçilen Modelin Yorumu

Elle bulduğumuz adayı şimdi otomatik bir aramayla karşılaştıralım.

```r
fit <- auto.arima(log(AirPassengers), seasonal = TRUE)
print(fit)
#> Series: log(AirPassengers)
#> ARIMA(0,1,1)(0,1,1)[12]
#>
#> Coefficients:
#>           ma1     sma1
#>       -0.4018  -0.5569
#> s.e.   0.0896   0.0731
#>
#> sigma^2 = 0.001371:  log likelihood = 244.7
#> AIC=-483.4   AICc=-483.21   BIC=-474.77
```

Bu blok, Box-Jenkins'in 1. ve 2. adımlarını (tanımlama ve kestirim) otomatik olarak yapar: Fark sayılarını belirler, aday modelleri dener, en iyisinin katsayılarını kestirir ve sonucu ekrana yazar. Kodu satır satır okuyalım:

1. `fit <- auto.arima(log(AirPassengers), seasonal = TRUE)`:
   - İlk argüman modele verilecek seridir. Log dönüşümünü kendimiz yapıp modele log seriyi veriyoruz; fark alma işlemlerini ise `auto.arima()` kendisi yapar. Bu yüzden `AP_stationary`'yi değil, farkı alınmamış log seriyi veririz.
   - Fonksiyon önce fark sayılarını 7.6.3'teki `nsdiffs()` ve `ndiffs()` ile belirler ($D = 1$, $d = 1$). Ardından farklı (p, q)(P, Q) kombinasyonlarını dener ve varsayılan `ic = "aicc"` ayarı gereği en küçük AICc değerine sahip modeli seçer (7.4).
   - `seasonal = TRUE` mevsimsel terimlerin de aranmasını sağlar (varsayılan zaten budur; `TRUE`/`FALSE` R'ın tırnaksız yazılan mantıksal değerleridir).
   - Kodda yazılmayan iki varsayılan argüman da önemlidir: `stepwise = TRUE`, bütün kombinasyonları denemek yerine iyi bir başlangıç modelinden komşu modellere adım adım ilerler; hızlıdır ama nadiren en iyi modeli kaçırabilir. `approximation` ise uzun serilerde (150'den fazla gözlem ya da 12'den büyük frekans) arama sırasında olabilirliği yaklaşık hesaplayarak zaman kazandırır; 144 gözlemli `AirPassengers` için zaten kapalıdır. Tam arama yapan `auto.arima(log(AirPassengers), stepwise = FALSE, approximation = FALSE)` de aynı modeli seçer.
   - Sonuç, `fit` adıyla saklanan bir **model nesnesidir**: İçinde seçilen dereceler, katsayılar, standart hatalar, artıklar ve bilgi kriterleri birlikte durur. Sonraki adımlarda (`checkresiduals()`, `forecast()`) hep bu nesneyi kullanacağız.
2. `print(fit)`: Model nesnesinin özetini ekrana yazar. Etkileşimli çalışırken yalnızca `fit` yazmak da aynı özeti verir; `print()` bunu açıkça ister.

Çıktıyı satır satır okuyalım:

- `Series: log(AirPassengers)`: Modelin hangi seriye kurulduğunu gösterir. Fonksiyona verdiğimiz ifade aynen yazılır; modelin log ölçekte olduğunu buradan da hatırlarız.
- `ARIMA(0,1,1)(0,1,1)[12]`: Seçilen model. İlk parantez (p, d, q) = (0, 1, 1) mevsimsel olmayan kısmı, ikinci parantez (P, D, Q) = (0, 1, 1) mevsimsel kısmı, köşeli parantezdeki 12 mevsim uzunluğunu verir (Şekil 7.3). `auto.arima()`, ACF/PACF'den elle çıkardığımız adayla aynı modeli, yani 7.3'teki **havayolu modelini** seçmiştir. İki parçanın anlamı aşağıda özetlenmiştir.
- `Coefficients:` başlığının altındaki küçük tablo kestirilen katsayılardır. Sütun adları katsayının hangi terime ait olduğunu söyler: `ma1` mevsimsel olmayan MA kısmının 1. katsayısı $\theta_1$, `sma1` mevsimsel (*seasonal*) MA kısmının 1. katsayısı $\Theta_1$'dir. Model AR terimi içermediği için `ar1` ya da `sar1` sütunu yoktur. Sabit terim (`drift`) sütunu da yoktur: $d + D = 2$ olduğu için `auto.arima()` sabit eklemez (7.2). İlk satır kestirimlerdir: $\theta_1 = -0.4018$, $\Theta_1 = -0.5569$. R, MA kısmını artı işaretiyle ($1 + \theta_1 B$) yazdığı için (7.1.2'deki not) bu sayılar 7.3'teki denkleme doğrudan konur.
- `s.e.` satırı her katsayının **standart hatasıdır** (7.5): Kestirimin ne kadar belirsiz olduğunu gösterir. Standart hatalar katsayıların çok altındadır: Katsayı/standart hata oranları $-0.4018 / 0.0896 \approx -4.5$ ve $-0.5569 / 0.0731 \approx -7.6$'dır. Kabaca "|katsayı| > 2 × s.e." kuralıyla iki katsayı da istatistiksel olarak anlamlıdır, yani sıfırdan belirgin biçimde farklıdır.
- `sigma^2 = 0.001371`: Beyaz gürültünün ($\varepsilon_t$) kestirilen varyansı $\sigma^2$'dir; `^` R'da üs işaretidir. Log ölçekte standart sapması $\sqrt{0.001371} \approx 0.037$ olduğundan (log farkı yaklaşık yüzde değişim olduğu için, 7.1.4) tek adımlık tipik hata yaklaşık **%3.7** mertebesindedir.
- `log likelihood = 244.7`: Modelin log olabilirliği $\log L$'dir (7.4); büyük olması veriye daha iyi uyum demektir.
- `AIC=-483.4   AICc=-483.21   BIC=-474.77`: Üç bilgi kriteri. AIC ile log likelihood arasındaki bağ 7.4'teki formüldür: $-2 \cdot 244.7 + 2 \cdot 3 = -483.4$ ($k = 3$: ma1, sma1 ve sigma^2). AICc, AIC'ye küçük örneklem düzeltmesi olarak $2k(k+1)/(n-k-1)$ ekler; burada $n = 131$'dir, çünkü iki fark işleminden sonra modelin kullanabildiği gözlem sayısı budur: $2 \cdot 3 \cdot 4 / (131 - 3 - 1) = 24/127 \approx 0.19$ ve $-483.40 + 0.19 = -483.21$. BIC cezada $2k$ yerine $k \log n$ kullanır: $-489.40 + 3 \cdot 4.875 \approx -474.77$. Üçü de "küçük olan daha iyi" kuralıyla okunur ve tek başına anlam taşımaz; **aynı veri üzerindeki** farklı modelleri karşılaştırmak için kullanılır (7.4'teki tablo).

> **Simge notu:** $`\approx`$ *(yaklaşık eşittir)*: iki değerin yaklaşık olarak eşit olduğunu belirtir · $`\sqrt{\cdot}`$ *(karekök)*: varyanstan standart sapmaya geçiş

**Sözlü özet:** `d = 1` ve `D = 1`, modelin aydan aya ve yıldan yıla değişime baktığını; `q = 1` ve `Q = 1`, tahminin geçen ayın ve geçen yılın aynı ayının sürprizleriyle düzeltildiğini; `p = P = 0` ise bunlardan sonra geçmiş değerlere ayrıca ihtiyaç kalmadığını söyler. Örneğin geçen Temmuz beklenenden yüksek çıktıysa mevsimsel fark bu sürprizi bu Temmuz'un tahminine olduğu gibi taşırdı; $\Theta_1 = -0.5569$ bunun yaklaşık %56'sını geri alır ve mevsimsel kısım üzerinden yalnızca yaklaşık %44'ü ($1 + \Theta_1$) taşınır. (Normal fark, sürprizin aradaki aylarda yükselttiği seviyeyi de taşıdığı için tahmine toplam etki sürprizin yaklaşık $2 + \theta_1 + \Theta_1 \approx 1.04$ katıdır.)

Şimdi bu modelin gerçekten işe yarayıp yaramadığını kontrol etmeliyiz.

#### 7.6.6. Teşhis: Artıkların İncelenmesi

Artıkların taşıması gereken üç özellik (ortalaması sıfır, varyansı sabit, otokorelasyonsuz) 7.4'te anlatıldı; şimdi bunları seçtiğimiz modelde denetleyelim.

```r
checkresiduals(fit)
#>  Ljung-Box test
#> data:  Residuals from ARIMA(0,1,1)(0,1,1)[12]
#> Q* = 26.446, df = 22, p-value = 0.233
#>
#> Model df: 2.   Total lags used: 24
```

Bu tek satır, Box-Jenkins'in 3. adımını (teşhis) yapar: Modelin artıklarının 7.4'teki özellikleri taşıyıp taşımadığını hem grafiklerle hem de 7.4'teki Ljung-Box testiyle denetler. Kodu okuyalım:

1. `checkresiduals(fit)`: `forecast` paketindeki bu fonksiyon, 7.6.5'te kurduğumuz `fit` model nesnesinin içinden artıkları alır ($e_t = x_t - \hat{x}_ t$; model log seriye kurulduğu için artıklar da log ölçektedir). Ardından iki iş yapar: Üç grafiği tek pencerede çizer (artıkların zaman grafiği, ACF'si ve histogramı) ve Ljung-Box testini uygulayıp sonucunu ekrana yazar. Testte kaç gecikmeye bakılacağını belirleyen `lag` argümanını vermediğimiz için onu kendisi seçer: Mevsimsel veride $2s$ ile $n/5$'ten küçük olanı alır; burada $\min(24,\ 144/5 = 28.8) = 24$'tür. Yalnızca test sonucu isteniyorsa `checkresiduals(fit, plot = FALSE)` grafikleri kapatır.

Çıktıyı satır satır okuyalım:

- `Ljung-Box test`: Yapılan testin adı. `data:  Residuals from ARIMA(0,1,1)(0,1,1)[12]` testin hangi modelin artıklarına uygulandığını söyler.
- `Q* = 26.446`: 7.4'teki $Q^{\ast}$ istatistiği; artıkların ilk 24 gecikmedeki otokorelasyonlarının karelerinden kurulan ağırlıklı toplam.
- `df = 22`: Testin serbestlik derecesi; kullanılan 24 gecikmeden kestirilen 2 katsayının düşülmesiyle elde edilir.
- `p-value = 0.233`: Ljung-Box p-değeri; aşağıda yorumluyoruz.
- `Model df: 2.`: Modelde kestirilen katsayı sayısı (ma1 ve sma1); $\sigma^2$ bu sayıya katılmaz.
- `Total lags used: 24`: Testte kullanılan gecikme sayısı, yani 7.4'teki formüldeki $h$.

Grafikler (Şekil 7.6) şöyle okunur:

- **Artıkların zaman grafiği:** Belirgin bir desen ya da trend olmamalı.
- **Artıkların ACF grafiği:** Çubukların neredeyse tamamı mavi kesikli güven bandı ($\pm 1.96/\sqrt{144} \approx \pm 0.16$) içinde kalmalı. 23. gecikmedeki bir çubuğun (yaklaşık 0.22) sınırı aşması, %5 anlamlılık düzeyinde tesadüfen beklenen bir durumdur: 24 çubuktan birinin ya da ikisinin bandı aşması normaldir.
- **Histogram:** Sıfır etrafında, normal dağılıma benzer bir şekil olmalı.

**Çıktının yorumu:** Artıklar saf gürültü olsaydı $Q^{\ast}$'ın ortalama değeri `df` kadar, yani 22 olurdu; bulunan 26.4 buna yakındır. 22 serbestlik dereceli ki-kare dağılımında %5 düzeyindeki sınır yaklaşık 33.9'dur ve 26.4 bunun altında kalır. Bu yüzden p-değeri 0.233'tür (> 0.05): "Artıklar arasında otokorelasyon yoktur" hipotezini reddedemiyoruz; artıklar beyaz gürültüden ayırt edilemiyor. Model teşhis aşamasını geçmiştir.

**Not —** Aynı test temel R'daki `Box.test()` ile de yapılabilir. Ancak bir modelin artıklarında, kestirilen katsayı sayısı `fitdf` argümanıyla verilmelidir:

```r
Box.test(residuals(fit), lag = 24, type = "Ljung-Box", fitdf = 2)
#> X-squared = 26.446, df = 22, p-value = 0.233
Box.test(residuals(fit), lag = 24, type = "Ljung-Box")   # fitdf unutulursa
#> X-squared = 26.446, df = 24, p-value = 0.3309
```

`residuals(fit)` model nesnesinden artıkları çeker; `fitdf = 2` (ma1 ve sma1) serbestlik derecesini 24'ten 22'ye indirir ve sonuç `checkresiduals()` ile aynı olur. `fitdf` unutulursa p-değeri olduğundan büyük çıkar (0.331) ve kötü bir model teşhisten gereğinden kolay geçebilir. 7.4b'deki kutuda `fitdf` gerekmemesinin nedeni, oradaki serinin bir modelin artıkları değil, saf bir seri olmasıdır.

![Artık teşhisi](../images/ch07_artik_teshisi.svg)

*Şekil 7.6 — ARIMA(0,1,1)(0,1,1)[12] modelinin gerçek artıkları: (a) zaman grafiği, (b) ACF ve $`\pm 1.96/\sqrt{n}`$ güven bandı, (c) normal eğriyle karşılaştırılan histogram.*

> **Hemen deneyin 7.6.6 · Kötü bir model teşhisten nasıl kalır?** (R, ~4 dk)
>
> Yalnızca fark alan, hiç MA terimi içermeyen bir model kurup artıklarına bakalım:
>
> ```r
> library(forecast)
> kotu <- Arima(log(AirPassengers), order = c(0, 1, 0), seasonal = c(0, 1, 0))   # yalnızca fark
> checkresiduals(kotu)
> ```
>
> `Arima()` dereceleri bizim verdiğimiz modeli kurar: `order` = (p, d, q), `seasonal` = (P, D, Q).
>
> **Ne göreceksiniz:** `Q* = 82.628, df = 24, p-value = 2.308e-08` ve `Model df: 0`. p-değeri sıfıra çok yakındır: Artıklarda otokorelasyon kalmıştır. Artıkların ACF grafiğinde lag-1 (yaklaşık −0.34) ve lag-12 (yaklaşık −0.39) çubukları bandı açıkça aşar. Bunlar 7.6.4'te `AP_stationary` için gördüğümüz çubuklardır: Bu modelin artıkları, farkı alınmış serinin neredeyse aynısıdır; modelin MA(1) ve mevsimsel MA(1) terimlerine ihtiyacı vardır.

#### 7.6.7. Tahmin (Öngörü)

Model teşhisten geçtiğine göre geleceği tahmin etmek için `forecast()` fonksiyonunu kullanabiliriz.

```r
# Gelecek 24 ay için tahmin (log ölçekte)
fc <- forecast(fit, h = 24)
plot(fc, main = "Gelecek 24 Ay için log(Yolcu Sayısı) Tahmini")
grid()

# Orijinal ölçeğe dönmek için exp() uygularız
round(exp(fc$mean[c(1, 12, 24)]), 1)   # 1., 12. ve 24. ay tahminleri
#> [1] 450.4 477.2 525.5
```

Bu blok, Box-Jenkins'in 4. adımını (öngörü) yapar: Teşhisten geçen modelle gelecek 24 ayı tahmin eder, sonucu çizer ve birkaç tahmini log ölçekten yolcu sayısına geri çevirir. Kodu satır satır okuyalım:

1. `fc <- forecast(fit, h = 24)`: `forecast()`, bir model nesnesini alıp geleceğe tahmin üretir. `h` (*horizon*, ufuk) kaç adım ileri gidileceğidir; veri aylık olduğu için 24 adım, Ocak 1961'den Aralık 1962'ye kadar 24 aydır. Tahmin aralıklarının düzeyini `level` argümanı belirler; yazmadığımız için varsayılan `c(80, 95)`, yani %80 ve %95 aralıkları hesaplanır. Model log seriye kurulduğu için tahminler de log ölçektedir. Sonuç, `fc` adıyla saklanan ve içinde birkaç parça barındıran bir `forecast` nesnesidir.
2. `plot(fc, main = "...")`: `plot()` bir `forecast` nesnesi aldığında ona özel çizim yöntemini kullanır: Geçmiş veriyi, nokta tahminlerini ve tahmin aralıklarını tek grafikte çizer. `main` başlıktır.
3. `grid()`: Açık grafiğe arka plan kılavuz çizgileri (ızgara) ekler; değerleri eksenlerden okumayı kolaylaştırır.
4. `round(exp(fc$mean[c(1, 12, 24)]), 1)`: İçten dışa okunur:
   - `fc$mean`: `$` işareti bir nesnenin içinden adıyla bir parça çeker. `fc$mean` 24 nokta tahmininden oluşan bir `ts` nesnesidir; `fc$lower` ve `fc$upper` ise aralıkların alt ve üst sınırlarını, %80 ve %95 için birer sütun olacak biçimde tutar.
   - `[c(1, 12, 24)]`: Köşeli parantez bir vektörden eleman seçer; içine verilen `c(1, 12, 24)` vektörü 1., 12. ve 24. elemanları, yani Ocak 1961, Aralık 1961 ve Aralık 1962 tahminlerini seçer. Bu tahminler log ölçektedir (örneğin ilk ay için 6.1102).
   - `exp()`: `log()`'un tersidir ($e^x$, $e \approx 2.718$) ve sayıyı orijinal ölçeğe döndürür: $e^{6.1102} \approx 450.4$.
   - `round(..., 1)`: Sonucu bir ondalığa yuvarlar.

   Bloktaki `#` ile başlayan satırlar ve satır sonundaki `#` sonrası yorumdur.

Ekrana yazılan `[1] 450.4 477.2 525.5`, üç elemanlı bir vektördür: Orijinal ölçekte nokta tahminleri Ocak 1961 için yaklaşık 450, Aralık 1961 için 477, Aralık 1962 için yaklaşık 526 bin yolcudur.

**Çıktının yorumu:** Grafikte mavi çizgi **nokta tahminlerini**, koyu ve açık gri alanlar sırasıyla **%80 ve %95 tahmin aralıklarını** gösterir. Model, öğrendiği trendi ve yıllık deseni geleceğe taşır; yaz tepeleri tahminlerde de görülür.

Gri alanların zamanla genişlemesi dikkat çekicidir: Ne kadar uzağı tahmin edersek belirsizlik o kadar artar. Bu, modelin uzun vadeli tahminlerde daha az kesin olduğunu dürüstçe ifade etmesidir. %95 aralığı Ocak 1961 için yaklaşık 419–484 iken Aralık 1962 için 400–691'e genişler (`exp(fc$lower)` ve `exp(fc$upper)` ile hesaplanır).

**Aralık nasıl hesaplanır?** Bir adım ilerisi için %95 aralığı, nokta tahmininin iki yanına 1.96 standart hata eklenerek bulunur (1.96 sayısının kaynağı Bölüm 6.3.5'tedir). Model log ölçekte kurulduğu için hesap log ölçekte yapılır. Ocak 1961 için nokta tahmini 6.1102, şokların standart sapması $\sqrt{0.001371} \approx 0.0370$'tir (7.6.5). Aralık $6.1102 \pm 1.96 \times 0.0370$, yani 6.0376 ile 6.1828 arasıdır. `exp()` ile geri çevrilince yaklaşık 418.9 ile 484.3 bin yolcu bulunur; Bölüm 1.3'teki "tahmin 450; gerçek değer %95 olasılıkla 420 ile 480 arasında" örneği bu sonuca yakındır. Ufuk uzadıkça her adımın belirsizliği bir sonrakine eklendiği için standart hata büyür: Aralık 1962 ($h = 24$) için log ölçekteki standart hata yaklaşık 0.1396'dır, yani bir adım ilerisinin yaklaşık 3.8 katıdır.

**Not —** Log ölçekteki tahminin `exp()` ile geri çevrilmesi, orijinal ölçekte ortalamayı değil **ortancayı** (medyanı) verir; ortalama tahmin bundan çok az daha yüksektir. Aralık 1962 ($h = 24$) için oran $e^{0.1396^2/2} \approx 1.01$'dir; yani ortalama 525.5 yerine yaklaşık 530.6 bin yolcudur. Bu ders kapsamında fark ihmal edilebilir düzeydedir.

**Not —** Bu tahminleri gerçek değerlerle karşılaştırıp modelin **başarısını sayısal olarak ölçmek** için veriyi eğitim ve test kısımlarına ayırmamız gerekir. Bunu, hata metrikleriyle birlikte Bölüm 8.3'te yapacağız.

> **Hemen deneyin 7.6 · Aynı akışı `USAccDeaths` ile tekrarlayın** (R, ~10 dk)
>
> `USAccDeaths`, ABD'de 1973–1978 arasındaki aylık kaza kaynaklı ölüm sayısıdır (72 gözlem). 7.6.1–7.6.7'deki adımları bu seride sırayla uygulayın: (1) çizip log gerekip gerekmediğine karar verin, (2) fark sayılarını bulup farkı alınmış serinin ACF'sine bakın, (3) modeli kurun, (4) artıkları denetleyin, (5) son yılı (1978) ayırıp tahmin hatasını ölçün.
>
> ```r
> library(forecast)
> plot(USAccDeaths)                                       # 1) dalga boyu büyüyor mu?
> round(tapply(USAccDeaths, cycle(USAccDeaths), mean))    #    her takvim ayının ortalaması
> nsdiffs(USAccDeaths); ndiffs(diff(USAccDeaths, lag = 12))   # 2) D ve d
> acf(diff(diff(USAccDeaths, lag = 12)), lag.max = 24)    #    59 gözlem
> fit_us <- auto.arima(USAccDeaths); fit_us               # 3) model
> checkresiduals(fit_us)                                  # 4) teşhis
> egitim <- window(USAccDeaths, end = c(1977, 12))        # 5) 1973-1977 eğitim
> test   <- window(USAccDeaths, start = c(1978, 1))       #    1978 test
> fc <- forecast(auto.arima(egitim), h = 12)
> round(accuracy(fc, test)[, c("MAE", "RMSE", "MAPE")], 2)
> ```
>
> `cycle()` her gözlemin yıl içindeki sırasını (1 = Ocak, …, 12 = Aralık) verir; `tapply(değerler, grup, mean)` her grubun ortalamasını alır. `accuracy(tahmin, test)` hata metriklerini eğitim ve test satırlarıyla verir (ayrıntısı Bölüm 8.2–8.3).
>
> **Ne göreceksiniz:**
>
> - Her yaz tekrar eden tepeler; en yüksek ortalama Temmuz (`10453`), en düşük Şubat (`7284`). Dalga boyu zamanla belirgin biçimde büyümez: Log gerekmez.
> - `nsdiffs` ve `ndiffs` ikisi de `[1] 1`: $D = 1$, $d = 1$. Güven bandı $\pm 1.96/\sqrt{59} \approx \pm 0.255$'tir; ACF'de yalnızca lag-1 (yaklaşık −0.35) ve lag-12 (yaklaşık −0.34) bandı aşar → aday ARIMA(0,1,1)(0,1,1)[12].
> - `auto.arima()` aynı havayolu modelini seçer: `ma1 = -0.4303` (s.e. 0.1228), `sma1 = -0.5528` (s.e. 0.1784), `sigma^2 = 102860`.
> - `Q* = 13.467, df = 12, p-value = 0.336`, `Total lags used: 14`: Model teşhisten geçer.
> - Test satırı: `MAE = 231.61`, `RMSE = 288.83`, `MAPE = 2.72`.
>
> **Şimdi siz:** (a) Katsayılar `AirPassengers`'takilere (−0.40, −0.56) ne kadar yakın, ikisi de anlamlı mı? `sigma^2` neden bu kadar büyük? (b) Ljung-Box neden 24 değil 14 gecikme kullandı?
>
> <details><summary>Cevap</summary>
>
> (a) Çok yakındır. −0.4303/0.1228 ≈ −3.5 ve −0.5528/0.1784 ≈ −3.1: "|katsayı| > 2 × s.e." kuralıyla ikisi de anlamlıdır. Standart hatalar daha büyüktür, çünkü seri yalnızca 72 gözlemdir. Log almadığımız için `sigma^2` ölüm sayısının karesi birimindedir. (b) 72 gözlemde gecikme sayısı $\min(24,\ 72/5 = 14.4)$, yani 14'tür; 2 katsayı düşülünce `df` 12 olur. Test MAE'si (231.61), mevsimsel naive'in 259.50'sinden (8.2.6'daki kutu) yalnızca yaklaşık %11 düşüktür: Bu seride "geçen yılı kopyala" zaten güçlü bir rakiptir.
>
> </details>

---

### 7.7. Python ile Aynı Analiz

> **Uygulama dosyası:** [`Codes/python/ch07_sarima_airpassengers.py`](../Codes/python/ch07_sarima_airpassengers.py) · [Notebook](../Codes/notebooks/ch07_sarima_airpassengers.ipynb) · [![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch07_sarima_airpassengers.ipynb)
>
> Bu bölümdeki kodların tamamı bu dosyada. Dosya, depo kök dizininde `python Codes/python/ch07_sarima_airpassengers.py` komutuyla ya da VS Code'da hücre hücre çalıştırılabilir. Kurulum yapmadan denemek için Colab bağlantısı kullanılabilir.


Aynı `AirPassengers` analizini şimdi Python ile, `statsmodels` ve `pmdarima` kütüphanelerini kullanarak yapalım. Burada tanımlayacağımız `data`, `train_data`, `test_data`, `predictions_arima` ve `rmse_arima` değişkenlerini sonraki bölümlerde, özellikle Bölüm 8.4'te ve Bölüm 15'teki model karşılaştırmasında, yeniden kullanacağız.

#### 7.7.1. Veri Setinin Yüklenmesi ve Hazırlanması

Önce gerekli kütüphaneleri projemize dahil edelim.

```python
# Gerekli kütüphaneleri içe aktarıyoruz.
# Kurulum: pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn
import pandas as pd  # Veri manipülasyonu ve analizi için temel kütüphane.
import numpy as np  # Sayısal hesaplamalar için temel kütüphane.
import matplotlib.pyplot as plt  # Veri görselleştirme için kullanılır.
from pmdarima.datasets import load_airpassengers  # AirPassengers veri setini yüklemek için.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ACF ve PACF grafikleri için
from statsmodels.tsa.stattools import adfuller, kpss  # ADF ve KPSS durağanlık testleri için
from pmdarima import auto_arima  # En uygun (S)ARIMA modelini otomatik bulmak için
from sklearn.metrics import mean_squared_error  # Ortalama kare hata (RMSE hesabı için)
```

`import pandas as pd` kütüphaneyi yükler ve ona kısa bir takma ad verir; bundan sonra `pandas.date_range` yerine `pd.date_range` yazarız. `from ... import ...` biçimi ise bir kütüphaneden yalnızca belirli fonksiyonları alır; örneğin `adfuller` ve `kpss` testleri doğrudan adlarıyla çağrılabilir hâle gelir. R'daki `library()` satırlarının Python'daki karşılığı bu satırlardır.

Şimdi veri setini yükleyelim. `pmdarima` veriyi 0, 1, 2, … şeklinde tam sayı indeksle getirir; grafiklerde yılları görebilmek için aylık bir tarih indeksi ekliyoruz.

```python
# AirPassengers veri setini pmdarima kütüphanesi yardımıyla yüklüyoruz.
# as_series=True parametresi ile veriyi bir Pandas Serisi olarak alıyoruz.
data = load_airpassengers(as_series=True)

# 1949 Ocak'tan başlayan aylık tarih indeksi ekleyelim ("MS" = ay başı).
data.index = pd.date_range(start="1949-01-01", periods=len(data), freq="MS")

# Verinin ilk beş satırını görüntüleyelim.
print(data.head())

# Veriyi görselleştirelim.
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Aylık Hava Yolu Yolcu Sayıları (1949-1960)')
plt.xlabel('Yıl')
plt.ylabel('Yolcu Sayısı')
plt.show()
```

`load_airpassengers(as_series=True)` 144 aylık değeri bir pandas `Series` nesnesi (her değerin bir etiketi, yani **indeksi** olan tek sütunluk veri) olarak getirir. `pd.date_range(start="1949-01-01", periods=len(data), freq="MS")` 1949 Ocak'tan başlayarak `len(data)` = 144 adet ay başı tarihi üretir; bu tarihleri `data.index`'e atayınca her değer kendi ayıyla etiketlenir. `data.head()` ilk beş satırı gösterir:

```text
1949-01-01    112.0
1949-02-01    118.0
1949-03-01    132.0
1949-04-01    129.0
1949-05-01    121.0
Freq: MS, dtype: float64
```

Grafik kısmında `plt.figure(figsize=(12, 6))` 12 × 6 inçlik bir çizim alanı açar, `plt.plot(data)` seriyi çizer (tarih indeksi yatay eksene yerleşir), `plt.title`, `plt.xlabel` ve `plt.ylabel` başlık ve eksen adlarını yazar, `plt.show()` grafiği ekranda gösterir. Grafik, R'da gördüğümüzün aynısıdır: artan bir trend, her yıl tekrarlanan mevsimsel dalgalanma ve zamanla büyüyen dalga boyu.

#### 7.7.2. Durağanlık Testleri ve ACF/PACF

```python
# ADF testi (H0: birim kök var, seri durağan değil)
adf_p = adfuller(data)[1]
# KPSS testi (H0: seri durağan)
kpss_p = kpss(data, regression='c', nlags='auto')[1]
print(f"Orijinal seri  -> ADF p = {adf_p:.3f}, KPSS p = {kpss_p:.3f}")

# Log + mevsimsel fark + normal fark
data_stationary = np.log(data).diff(12).diff().dropna()
print(f"Durağanlaştırılmış seri -> ADF p = {adfuller(data_stationary)[1]:.4f}, "
      f"KPSS p = {kpss(data_stationary, regression='c', nlags='auto')[1]:.3f}")
```

```text
Orijinal seri  -> ADF p = 0.992, KPSS p = 0.010
Durağanlaştırılmış seri -> ADF p = 0.0002, KPSS p = 0.100
```

`adfuller(data)` tek bir sayı değil, bir **demet** (*tuple*: parantez içinde sıralanmış, değiştirilemeyen bir değer listesi) döndürür. Sırasıyla şunları içerir: test istatistiği, p-değeri, kullanılan gecikme sayısı, regresyondaki gözlem sayısı, kritik değerler sözlüğü (`{'1%': ...}` biçiminde anahtar–değer çiftleri) ve gecikme seçiminde kullanılan en iyi bilgi kriteri değeri. Orijinal seri için bu demet yaklaşık olarak (0.815, 0.992, 13, 130, {'1%': −3.48, '5%': −2.88, '10%': −2.58}, …) biçimindedir. Python'da sıra numarası 0'dan başladığı için `[1]` ikinci elemanı, yani p-değerini alır. `kpss(data, regression='c', nlags='auto')` de benzer bir demet döndürür (istatistik, p-değeri, gecikme sayısı, kritik değerler); `regression='c'` seviye durağanlığını sınar (R'daki `null = "Level"`), `nlags='auto'` gecikme sayısını veriden seçer. `print(f"...")` satırı bir *f-dizesi* (f-string) kullanır: süslü parantez içindeki değer metne yerleştirilir; `:.3f` onu üç ondalık basamakla gösterir.

`np.log(data).diff(12).diff().dropna()` satırı R'daki `diff(diff(log(...), lag = 12))` ile aynı işi soldan sağa zincir hâlinde yapar: log al, 12 aylık farkı al, 1 aylık farkı al. Fark alınamayan ilk 13 ay boş değer (`NaN`) olarak kalır; `.dropna()` bunları atar.

**Çıktının yorumu:** Orijinal seride ADF p ≈ 0.99, KPSS p = 0.01: İki test de **durağan değil** diyor. R'daki çelişki burada görülmez; bunun iki nedeni vardır. Birincisi, `adfuller()` varsayılan olarak regresyona trend koymaz, yalnızca sabit kullanır (`regression='c'`); R'daki `adf.test()` ise her zaman sabit ve trend kullanır. İkincisi, `adfuller()` gecikme sayısını AIC ile seçer (`autolag='AIC'`) ve burada 13 gecikme seçerek 12 aylık hafızayı kapsar. Yine de sonuç, regresyonda trend olmadığı için 7.6.2'deki `k = 13` sonucunun (sabit + trend, −2.10) karşılığı değildir: Yalnızca sabitli ve 13 gecikmeli regresyonun istatistiği 0.815'tir (R'da `urca::ur.df(x, type = "drift", lags = 13)` de 0.8154 verir). Ayrıca `adfuller()` p-değerini sürekli bir formülle (MacKinnon) hesapladığı için 0.01–0.99 aralığına sıkıştırmaz. Aşağıdaki kutuda R'ın −7.3186'sı, ayarlar eşitlenerek yeniden üretilir. Durağanlaştırılmış seride ADF p ≈ 0.0002, KPSS p ≥ 0.1: seri durağandır.

**Not —** KPSS p-değeri tablo sınırlarının dışında kaldığında `statsmodels` bir `InterpolationWarning` uyarısı verir; bu, gerçek p-değerinin gösterilenden daha küçük (ya da daha büyük) olduğunu belirtir, bir hata değildir. `statsmodels`'ın yeni sürümleri (0.15 ve sonrası) `adfuller` ve `kpss` için ayrıca bir `FutureWarning` yazabilir: İleride bu fonksiyonlar demet yerine bir sonuç nesnesi döndürecektir. Bu uyarı bugünkü sonuçları değiştirmez.

Şimdi ACF ve PACF grafiklerini çizelim:

```python
# ACF ve PACF grafiklerini çizdirelim
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Orijinal veri için ACF grafiği
plot_acf(data, ax=ax1, lags=40)
ax1.set_title('Otokorelasyon Fonksiyonu (ACF)')

# Orijinal veri için PACF grafiği
plot_pacf(data, ax=ax2, lags=40)
ax2.set_title('Kısmi Otokorelasyon Fonksiyonu (PACF)')

plt.tight_layout()
plt.show()
```

`plt.subplots(2, 1, ...)` alt alta iki grafik alanı (2 satır, 1 sütun) oluşturur ve bunları `ax1`, `ax2` adlarıyla geri verir; R'daki `par(mfrow = ...)` ile aynı amaca hizmet eder. `plot_acf(data, ax=ax1, lags=40)` ACF'yi ilk alana 40 gecikmeye kadar çizer; `ax=` argümanı grafiğin hangi alana çizileceğini söyler. `plt.tight_layout()` başlıkların ve eksen yazılarının üst üste binmemesi için boşlukları ayarlar.

**Çıktının yorumu:** Orijinal verinin ACF'si çok yavaş azalır ve 12, 24, 36. gecikmelerde tümsekler yapar. Yavaş sönüm trendin (durağan olmamanın), tümsekler mevsimselliğin işaretidir. Durağanlaştırılmış seri (`data_stationary`) aynı fonksiyonlarla çizildiğinde (uygulama dosyasında bu hücre de vardır), 7.6.4'te R'da gördüğümüz gecikme 1 ve 12'deki negatif çubuklar görülür.

> **Hemen deneyin 7.7.2 · R'ın ADF sonucunu Python'da yeniden üretin** (Python, ~3 dk)
>
> R'daki −7.3186 sonucu, `adf.test()`'in varsayılanları olan "sabit + trend, 5 gecikme" ayarlarından gelir. Aynı ayarları Python'da verelim:
>
> ```python
> import pandas as pd
> from statsmodels.tsa.stattools import adfuller
>
> seri = pd.read_csv("data/AirPassengers.csv", index_col="Month", parse_dates=True)["Passengers"]
> sonuc = adfuller(seri, regression="ct", maxlag=5, autolag=None)   # R'daki adf.test ayarları
> print(round(sonuc[0], 4), sonuc[1])                                # istatistik ve p-değeri
> ```
>
> `regression="ct"` sabit ve trend (*constant + trend*) ekler; `autolag=None` gecikme sayısını seçtirmez, `maxlag=5`'i aynen kullanır.
>
> **Ne göreceksiniz:** `-7.3186 2.658978691224247e-09`. İstatistik R ile aynıdır. p-değeri ise 0.0000000027'dir: R bunu tablonun ucunda 0.01'e sabitleyip "smaller than printed" uyarısı verirken Python sürekli formülle gerçek değeri yazar. `statsmodels` 0.15 ve sonrasında bir `FutureWarning` da görebilirsiniz; sonucu etkilemez.
>
> **Şimdi siz:** `maxlag=13` ile tekrarlayın. R'daki `adf.test(AirPassengers, k = 13)` sonucuyla (−2.1008, p = 0.5345) karşılaştırın.
>
> <details><summary>Cevap</summary>
>
> `-2.1008 0.5456589343124554`. İstatistik birebir aynı; p-değeri biraz farklıdır (0.546 ve 0.535), çünkü iki yazılım p-değerini farklı yöntemle hesaplar. Karar aynıdır: birim kök reddedilmez.
>
> </details>

#### 7.7.3. Veriyi Eğitim ve Test Olarak Ayırma

Modelin performansını ölçmek için verinin son 5 yılını (60 ay) test seti, geri kalanını eğitim seti olarak ayıralım. Zaman serisinde bu ayrım **rastgele değil, kronolojik** yapılır; nedenini Bölüm 8.1'de ayrıntılı olarak ele alacağız.

```python
# Veri setini eğitim ve test olarak ayırıyoruz. Son 60 ay test verisi olacak.
train_data = data[:-60]
test_data = data[-60:]
print(f"Eğitim: {train_data.index[0]:%Y-%m} - {train_data.index[-1]:%Y-%m} ({len(train_data)} ay)")
print(f"Test:   {test_data.index[0]:%Y-%m} - {test_data.index[-1]:%Y-%m} ({len(test_data)} ay)")
```

```text
Eğitim: 1949-01 - 1955-12 (84 ay)
Test:   1956-01 - 1960-12 (60 ay)
```

Köşeli parantez içindeki iki nokta (`:`) bir aralık seçer ve eksi sayılar sondan sayar: `data[:-60]` "baştan, sondan 60. elemana kadar (o hariç)", `data[-60:]` ise "sondan 60. elemandan sona kadar" demektir. `train_data.index[0]` ilk tarihi, `index[-1]` son tarihi verir; f-dizesindeki `:%Y-%m` tarihi "yıl-ay" biçiminde yazar.

**Not —** R uygulamasında (Bölüm 8.3) test seti olarak yalnızca 1960 yılını (12 ay) kullanacağız. Burada 60 ay seçmemizin nedeni, Bölüm 15'teki derin öğrenme modelleriyle (LSTM, GRU, 1D-CNN) aynı test dönemi üzerinde karşılaştırma yapabilmektir. Test ufku uzadıkça hata büyür; iki uygulamanın hata değerleri bu yüzden doğrudan karşılaştırılamaz.

#### 7.7.4. `auto_arima` ile En Uygun Modeli Bulma

(p, d, q) ve mevsimsel (P, D, Q, m) parametrelerini elle belirlemek yerine, R'daki `auto.arima()`'nın Python karşılığı olan `auto_arima` fonksiyonunu kullanabiliriz. Fonksiyon önce fark sayılarını testlerle belirler (varsayılan olarak mevsimsel fark $D$ için OCSB testi, yani mevsimsel fark gerekip gerekmediğini sınayan test; normal fark $d$ için KPSS testi), sonra farklı parametre kombinasyonlarını deneyerek en düşük AIC değerine sahip modeli bulur. `pmdarima`'da mevsim uzunluğu `s` yerine `m` parametresiyle verilir.

```python
# auto_arima fonksiyonunu kullanarak en uygun ARIMA modelini buluyoruz.
# seasonal=True, veride mevsimsellik olduğunu belirtir.
# m=12, mevsimsel döngünün 12 ay olduğunu (yıllık) ifade eder.
# stepwise=True, tüm kombinasyonlar yerine daha hızlı bir adımsal arama yapar.
# trace=True, denenen her modeli ve AIC değerini ekrana yazar.
auto_model = auto_arima(train_data,
                        seasonal=True,
                        m=12,
                        stepwise=True,
                        suppress_warnings=True,
                        trace=True)

# Bulunan en iyi modelin özetini yazdırıyoruz.
print(auto_model.summary())
```

Argümanlar kod içindeki yorumlarda özetlenmiştir: `seasonal=True` ve `m=12` mevsimsel terimlerin 12 aylık döngüyle aranmasını, `stepwise=True` R'daki gibi adımsal aramayı, `trace=True` her denemenin ekrana yazılmasını sağlar. `suppress_warnings=True` ise bazı aday modellerin kurulumu sırasında çıkan yakınsama uyarılarını gizler; bu uyarılar seçilen modeli etkilemez. Model yalnızca `train_data` ile kurulur; test dönemi hiç görülmez. Çıktının kısaltılmış hâli şöyledir:

```text
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(1,1,1)[12] intercept   : AIC=542.053, Time=2.55 sec
 ...
 ARIMA(1,0,0)(0,1,1)[12] intercept   : AIC=536.644, Time=0.41 sec
 ...
Best model:  ARIMA(1,0,0)(0,1,1)[12] intercept
...
Model:             SARIMAX(1, 0, 0)x(0, 1, [1], 12)   Log Likelihood                -264.322
...
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      6.4279      2.678      2.401      0.016       1.180      11.676
ar.L1          0.7530      0.103      7.346      0.000       0.552       0.954
ma.S.L12      -0.2999      0.150     -1.997      0.046      -0.594      -0.006
sigma2        87.9963     15.477      5.686      0.000      57.662     118.331
...
Ljung-Box (L1) (Q):                   0.58   Jarque-Bera (JB):                 3.41
Prob(Q):                              0.45   Prob(JB):                         0.18
```

**Çıktının yorumu:** `trace=True` sayesinde denenen modeller ve AIC değerleri satır satır listelenir (süreler bilgisayara göre değişir); en sonda seçilen model `SARIMAX(p,d,q)x(P,D,Q,12)` biçiminde raporlanır. Özet tablosundaki `ar.L1`, `ma.L1`, `ma.S.L12` gibi satırlar sırasıyla $\phi_1$, $\theta_1$, $\Theta_1$ katsayılarıdır; `intercept` sabit terim, `sigma2` beyaz gürültünün varyansıdır. `P>|z|` sütunu her katsayı için "bu katsayı aslında sıfır mı?" testinin p-değeridir; küçük değerler katsayının anlamlı olduğunu gösterir (`ma.S.L12` için 0.046, sınırda anlamlı). Tablonun altındaki `Ljung-Box (L1) (Q)` satırı, 7.6.6'daki artık testinin **yalnızca 1. gecikme** için yapılmış hâlidir (`Prob(Q)` = 0.45); R'ın 24 gecikmeye bakan testinden daha dar bir kontroldür. Yanındaki `Jarque-Bera (JB)` ise artıkların normal dağılıma uyup uymadığını çarpıklık ve basıklık üzerinden sınayan testtir; `Prob(JB)` = 0.18 olduğu için normallik reddedilmez.

Bizim denememizde (pmdarima 2.1) seçilen model sabit terimli **ARIMA(1,0,0)(0,1,1)[12]** oldu. Burada $d = 0$ olması şaşırtıcı değildir: Mevsimsel fark ($D = 1$) trendin büyük kısmını zaten temizlemiş, kalan kayma sabit terimle karşılanmıştır. Fark alınmış bir seride sabit terim seviyeyi değil, ortalama değişimi belirler (7.2'deki *drift*); burada $d + D = 1$ olduğu için sabit terime izin verilir. Modeli $u_t = x_t - x_{t-12}$ (bu ay eksi geçen yılın aynı ayı; 7.3'teki $w_t$'den farklı olarak log ve normal fark yoktur) cinsinden yazarsak $u_t = 6.43 + 0.753\thinspace  u_{t-1} + \varepsilon_t - 0.300\thinspace  \varepsilon_{t-12}$ olur. 7.1.1'deki ortalama formülüyle $u_t$'nin uzun dönem ortalaması $6.43 / (1 - 0.753) \approx 26.0$'dır: Model, her ayın geçen yılın aynı ayından ortalama yaklaşık 26 bin yolcu fazla olacağını öğrenmiştir (eğitim dönemindeki gerçek ortalama 26.2'dir). R'daki modelden farklı olmasının iki nedeni vardır: Burada log dönüşümü yapmadık ve model yalnızca 1956 öncesi veriyle kuruldu. Kütüphane sürümüne göre sonuç biraz farklı olabilir.

#### 7.7.5. Tahmin ve Değerlendirme

Şimdi bulduğumuz modelle test dönemi için tahmin yapalım ve gerçek değerlerle karşılaştıralım.

```python
# Test seti için tahminler yapıyoruz. n_periods, tahmin edilecek dönem sayısını belirtir.
predictions_arima = auto_model.predict(n_periods=len(test_data))

# Tahminleri, test verisi ile aynı indekse sahip bir Pandas Serisine dönüştürelim.
predictions_arima = pd.Series(np.asarray(predictions_arima), index=test_data.index)

# Gerçek değerler ve tahminleri görselleştirelim.
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Eğitim Verisi')
plt.plot(test_data, label='Gerçek Değerler (Test)', color='orange')
plt.plot(predictions_arima, label='ARIMA Tahminleri', color='green')
plt.title('ARIMA Modeli ile Yolcu Sayısı Tahmini')
plt.xlabel('Yıl')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()

# Modelin performansını Kök Ortalama Kare Hata (RMSE) ile ölçelim.
rmse_arima = np.sqrt(mean_squared_error(test_data, predictions_arima))
print(f'ARIMA Modeli RMSE Değeri: {rmse_arima:.2f}')
```

```text
ARIMA Modeli RMSE Değeri: 47.87
```

`auto_model.predict(n_periods=len(test_data))` eğitim döneminin sonundan itibaren 60 ay ileriye tahmin üretir (R'daki `forecast(fit, h = 60)` karşılığı). `np.asarray(...)` sonucu düz bir sayı dizisine çevirir; `pd.Series(..., index=test_data.index)` bu diziyi test dönemiyle aynı tarih etiketlerine sahip bir seriye dönüştürür, böylece gerçek değerlerle aynı grafikte doğru aylara oturur. `label=` her çizginin adını, `plt.legend()` bu adları gösteren açıklama kutusunu ekler. Son iki satırda `mean_squared_error(gerçek, tahmin)` hataların karelerinin ortalamasını hesaplar, `np.sqrt` karekökünü alarak sonucu yeniden yolcu birimine döndürür; bu, **RMSE**'dir.

**Çıktının yorumu:** Grafikte yeşil tahmin çizgisinin trendi ve yaz tepelerini genel olarak izlediği, ancak gerçek değerlerin giderek daha fazla altında kaldığı görülür: Tahmin hataları hep aynı yöndedir ve yıllık ortalama hata 1956'da yaklaşık 18 bin yolcu iken 1960'ta 60'ı aşar (yaklaşık 63). Nedeni 7.7.4'teki model yapısından okunabilir: Model her ayın geçen yılın aynı ayından yaklaşık 26 bin yolcu fazla olacağını varsayar, oysa test döneminde bu yıllık artış ortalama 38.4 olmuştur. Yolcu sayısı yüzde olarak benzer hızla büyüdükçe artışın yolcu cinsinden büyüklüğü de büyür; sabit bir yıllık artış varsayan model geride kalır. RMSE (yaklaşık 47.9 bin yolcu), 60 aylık uzun bir ufukta ortalama hatanın büyüklüğünü kabaca özetler.

**Not —** Burada kullandığımız **RMSE** (Root Mean Squared Error) ve diğer hata metrikleri (MAE, MAPE) bir sonraki bölümde, Bölüm 8.2'de formülleri ve yorumlarıyla ayrıntılı olarak tanımlanacaktır. Şimdilik "küçük RMSE = daha iyi tahmin" demek yeterlidir.

> **Hemen deneyin 7.7.5 · ARIMA'yı basit bir rakiple karşılaştırın** (Python, ~5 dk)
>
> Bir modelin RMSE'si tek başına anlamsızdır; onu en basit tahmin yöntemiyle karşılaştırmak gerekir. **Mevsimsel naif** tahmin şunu der: "Gelecek her yıl, eğitimin son yılının aynısı olacak."
>
> ```python
> import numpy as np
> import pandas as pd
>
> seri = pd.read_csv("data/AirPassengers.csv", index_col="Month", parse_dates=True)["Passengers"]
> train, test = seri[:-60], seri[-60:]
> naif = np.tile(train[-12:].values, 5)          # son 12 ayı 5 kez arka arkaya diz (60 ay)
> rmse = np.sqrt(np.mean((test.values - naif) ** 2))
> print(round(rmse, 2))
> ```
>
> `np.tile(dizi, 5)` diziyi 5 kez arka arkaya ekler. RMSE'yi `sklearn` yerine doğrudan formülle hesapladık: hataların karelerinin ortalamasının karekökü.
>
> **Ne göreceksiniz:** `126.01`. ARIMA'nın 47.87'lik RMSE'si bundan çok daha iyidir: Naif yöntem büyümeyi hiç hesaba katmaz.
>
> **Şimdi siz:** Naif tahmine büyümeyi ekleyin: Eğitimin son yılındaki ortalama yıllık artışı hesaplayıp her yıl üst üste ekleyin.
>
> ```python
> artis = (train[-12:].values - train[-24:-12].values).mean()   # 1955'in 1954'e göre aylık ortalama artışı
> naif2 = naif + np.repeat(artis * np.arange(1, 6), 12)           # 1. yıl +1 artış, 2. yıl +2 artış, ...
> print(round(artis, 2), round(np.sqrt(np.mean((test.values - naif2) ** 2)), 2))
> ```
>
> Sonuç ARIMA'dan iyi mi, kötü mü?
>
> <details><summary>Cevap</summary>
>
> `45.08 37.53`. Bu basit "geçen yıl + sabit artış" kuralı, bu test döneminde ARIMA'nın 47.87'lik RMSE'sinden **daha iyidir**. Nedeni yukarıdaki yorumda açıklanmıştı: `auto_arima`'nın seçtiği model yıllık artışı yaklaşık 26 öğrenmişti, oysa 1955'teki artış 45'tir. Ders: Karmaşık bir modeli her zaman basit bir referans yöntemle karşılaştırın (Bölüm 8).
>
> </details>

**Alıştırma:** Modeli `np.log(train_data)` üzerinde kurun, tahminleri `np.exp()` ile orijinal ölçeğe döndürün ve RMSE'yi yeniden hesaplayın. Bizim denememizde `auto_arima` log seride ARIMA(2,0,0)(0,1,1)[12] modelini seçti ve RMSE yaklaşık 49.0 çıktı; yani bu bölmede toplam hata azalmadı. Hataları yıllara göre inceleyin: Log model 1956–1957'de neredeyse isabetlidir (yıllık ortalama hata 3 bin yolcunun altında), 1958'den sonra ise fazla tahmin yapar, çünkü eğitim döneminin yüksek yüzde büyümesini sürdürürken gerçek büyüme 1958'de yaklaşık %3'e düşmüştür. Bu alıştırma, kuramsal olarak doğru bir dönüşümün her test döneminde daha düşük hata garanti etmediğini ve modelleri her zaman test verisi üzerinde karşılaştırmak gerektiğini gösterir.

---

### 7.8. Üstel Düzeltme (ETS) ve Theta

ARIMA bir seriyi, geçmiş değerleri ve geçmiş şokları arasındaki korelasyon yapısıyla anlatır. Bu bölümde aynı probleme başka bir kapıdan gireceğiz: Seriyi Bölüm 2.2'deki bileşenlerine (düzey, trend, mevsimsellik) ayırıp her yeni gözlem geldiğinde bu bileşenleri **biraz** güncellemek. Bu fikre **üstel düzeltme** (*exponential smoothing*) denir. 1950'lerin sonunda Brown, Holt ve Winters'ın çalışmalarıyla doğan bu yöntemlerin hesabı çok basittir, ama tahmin yarışmalarında ARIMA ile omuz omuza yarışırlar (Bölüm 17.10). Bölüm 5.3'teki tabloda adı geçen `HoltWinters` da bu ailedendir. 7.8.1–7.8.3'te yöntemi parça parça kuracağız (yalnızca düzey; düzey ve trend; düzey, trend ve mevsim), 7.8.4'te hepsini tek bir istatistiksel çerçevede (ETS) toplayacağız, 7.8.5'te yakın akrabası Theta yöntemini tanıyacağız ve 7.8.6'da hepsini `AirPassengers` üzerinde SARIMA ile karşılaştıracağız.

#### 7.8.1. Basit Üstel Düzeltme (SES)

**Açıklama:** Yarının değerini tahmin etmenin iki uç yolu vardır. **Naive** yöntem yalnızca son gözleme bakar: "Yarın bugün gibi olacak." **Ortalama** yöntemi ise bütün geçmişe eşit ağırlık verir: Elli gün önceki gözlem de dünkü kadar önemlidir. Birincisi gürültüye fazla duyarlıdır, ikincisi değişime fazla kördür. Basit üstel düzeltme ikisinin arasında durur: Bütün geçmişi kullanır, ama yakın geçmişe daha çok, uzak geçmişe giderek daha az ağırlık verir.

**Tanım:** Basit üstel düzeltme (*simple exponential smoothing*, SES), serinin o anki **düzeyini** (*level*) $\ell_t$ her yeni gözlemde şöyle günceller ve tahmini bu düzeye eşitler:

```math
\underbrace{\ell_t}_{\text{yeni düzey}} = \underbrace{\alpha\, x_t}_{\substack{\text{yeni gözlemin}\\\text{payı}}} + \underbrace{(1 - \alpha)\, \ell_{t-1}}_{\substack{\text{eski düzeyin}\\\text{payı}}}, \qquad \underbrace{\hat{x}_{t+h|t} = \ell_t}_{\substack{\text{her ufuk için}\\\text{aynı tahmin}}}
```

**Ne anlatıyor?** Yeni düzey, bugünkü gözlem ile dünkü düzeyin ağırlıklı ortalamasıdır; ağırlıkları düzeltme katsayısı $\alpha$ belirler. Kaç adım ileriye bakılırsa bakılsın tahmin son düzeydir, yani tahmin grafiği yatay bir çizgidir.

> **Simge notu:** $`\ell_t`$ *(el; el yazısı "l")*: $`t`$ anındaki düzey · $`\alpha`$ *(alfa)*: düzeltme katsayısı (*smoothing parameter*), 0 ile 1 arasında · $`\hat{x}_{t+h|t}`$ *(x şapka)*: $`t`$ anına kadarki bilgiyle $`h`$ adım sonrası için yapılan tahmin; dikey çizgi "… bilindiğinde" diye okunur

Aynı denklem $\ell_t = \ell_{t-1} + \alpha\,(x_t - \ell_{t-1})$ biçiminde de yazılabilir. Parantez içi, dünkü tahminin bugün yaptığı hatadır: Yeni düzey, eski tahmine bu hatanın $\alpha$ kadarı eklenerek bulunur. $\alpha = 0.3$ ise model her sürprizin %30'unu kalıcı sayar, %70'ini gürültü kabul eder.

**"Üstel" adı nereden geliyor?** Denklemde $\ell_{t-1}$ yerine kendi formülünü, onun içindeki $\ell_{t-2}$ yerine yine kendi formülünü koymayı sürdürürsek düzey, geçmiş gözlemlerin ağırlıklı toplamına açılır:

```math
\ell_t = \underbrace{\alpha\, x_t}_{\text{bugün}} + \underbrace{\alpha (1 - \alpha)\, x_{t-1}}_{\text{dün}} + \underbrace{\alpha (1 - \alpha)^2\, x_{t-2}}_{\text{evvelsi gün}} + \dots + \underbrace{\alpha (1 - \alpha)^k\, x_{t-k}}_{k\text{ adım önce}} + \dots
```

**Ne anlatıyor?** $k$ adım önceki gözlemin ağırlığı $\alpha (1 - \alpha)^k$'dir: Geriye doğru her adımda ağırlık $(1 - \alpha)$ ile çarpılarak küçülür. Bu geometrik (üstel) sönüm yönteme adını verir; ağırlıkların toplamı 1'e yaklaşır.

$\alpha = 0.3$ için ağırlıklar en yeniden geriye doğru 0.300, 0.210, 0.147, 0.103, … biçiminde yavaş azalır; $\alpha = 0.8$ için 0.800, 0.160, 0.032, … biçiminde hızla sıfıra iner. İki uç durum baştaki iki yöntemi verir: $\alpha = 1$ iken ağırlığın tamamı son gözleme gider ve SES naive yönteme dönüşür; $\alpha$ sıfıra yaklaştıkça ağırlıklar çok yavaş söner ve tahmin uzun dönem ortalamasına yaklaşır.

**Sayısal örnek.** Bir kırtasiyenin dört günlük defter satışları 40, 46, 43 ve 49 olsun. $\alpha = 0.3$ alalım ve başlangıç düzeyini ilk gözleme eşitleyelim ($\ell_1 = x_1 = 40$):

| Gün $`t`$ | Satış $`x_t`$ | Hesap: $`0.3\, x_t + 0.7\, \ell_{t-1}`$ | Düzey $`\ell_t`$ |
| --- | --- | --- | --- |
| 1 | 40 | başlangıç | 40 |
| 2 | 46 | $`0.3 \cdot 46 + 0.7 \cdot 40 = 13.8 + 28`$ | 41.8 |
| 3 | 43 | $`0.3 \cdot 43 + 0.7 \cdot 41.8 = 12.9 + 29.26`$ | 42.16 |
| 4 | 49 | $`0.3 \cdot 49 + 0.7 \cdot 42.16 = 14.7 + 29.512`$ | 44.212 |

Beşinci gün için tahmin $\hat{x}_ {5|4} = 44.21$'dir; altıncı ve yedinci gün için de aynıdır. Satışlar dört günde 40'tan 49'a çıkmış, ama tahmin 44.2'de kalmıştır: SES yükselişi kalıcı bir eğilim olarak değil, düzeyin biraz yukarı kayması olarak görür. Bu yüzden SES yalnızca belirgin trendi ve mevsimselliği olmayan seriler için uygundur; trendli bir seride hep geride kalır. 7.8.2'nin konusu budur.

Gerçek uygulamada $\alpha$ ve başlangıç düzeyi elle seçilmez: Yazılım bunları, bir adım ileri tahmin hatalarının karelerinin toplamını en küçük ya da olabilirliği en büyük yapacak biçimde veriden **kestirir** (7.4).

> **Hemen deneyin 7.8.1 · $`\alpha`$'yı değiştirip ağırlıkları izleyin** (R, ~3 dk)
>
> Yukarıdaki tabloyu bir döngüyle yeniden üretelim:
>
> ```r
> x <- c(40, 46, 43, 49)               # dört günlük satış
> alfa <- 0.3
> duzey <- x[1]                        # başlangıç düzeyi: ilk gözlem
> for (t in 2:4) {
>   duzey <- alfa * x[t] + (1 - alfa) * duzey
>   cat(t, round(duzey, 3), "\n")
> }
> round(alfa * (1 - alfa)^(0:5), 3)    # en yeni gözlemden geriye doğru ağırlıklar
> ```
>
> `for (t in 2:4) { ... }` süslü parantezin içini `t` = 2, 3, 4 için sırayla çalıştırır; her turda `duzey` bir öncekinin üzerine yazılır. `0:5` 0'dan 5'e kadar tam sayılardır ve `^` her birini üs olarak kullanır.
>
> **Ne göreceksiniz:** `2 41.8`, `3 42.16`, `4 44.212` ve `[1] 0.300 0.210 0.147 0.103 0.072 0.050`: tablodaki düzeyler ve $`\alpha = 0.3`$ ağırlıkları. `forecast` paketindeki `ses(x, alpha = 0.3, initial = "simple", h = 2)$mean` aynı tahmini (`44.212 44.212`) verir.
>
> **Şimdi siz:** `alfa <- 0.8` yapıp tekrar çalıştırın. Düzeyler gözlemlere daha mı yakın? Kaç adım önceki gözlemin ağırlığı 0.01'in altına düşer?
>
> <details><summary>Cevap</summary>
>
> `2 44.8`, `3 43.36`, `4 47.872`: Düzey her gün son gözleme çok yaklaşır (49'a karşı 47.87); tahmin artık naive yönteme benzer. Ağırlıklar `0.800 0.160 0.032 0.006 0.001 0.000` olur: 3 adım önceki gözlemin ağırlığı (0.006) bile 0.01'in altındadır. Büyük $`\alpha`$ kısa hafıza, küçük $`\alpha`$ uzun hafıza demektir.
>
> </details>

#### 7.8.2. Holt Yöntemi: Trend ve Sönümlü Trend

**Açıklama:** Holt (1957), SES'e ikinci bir bileşen ekledi: serinin her adımda ne kadar arttığını gösteren **eğim** (*slope*) $b_t$. Artık iki şey güncellenir, düzey ve eğim; her birinin kendi düzeltme katsayısı vardır.

```math
\begin{aligned}
\underbrace{\ell_t}_{\text{düzey}} &= \alpha\, x_t + (1 - \alpha)\, \underbrace{(\ell_{t-1} + b_{t-1})}_{\substack{\text{dünkü bilgiyle}\\\text{bugünün tahmini}}} \\
\underbrace{b_t}_{\text{eğim}} &= \beta\, \underbrace{(\ell_t - \ell_{t-1})}_{\substack{\text{düzeydeki}\\\text{son değişim}}} + (1 - \beta)\, b_{t-1} \\
\underbrace{\hat{x}_{t+h|t}}_{\text{tahmin}} &= \ell_t + \underbrace{h\, b_t}_{h\text{ adımlık artış}}
\end{aligned}
```

**Ne anlatıyor?** İç içe iki SES vardır. Düzey, bugünkü gözlem ile "dünkü düzey artı bir adımlık eğim" tahmininin ağırlıklı ortalamasıdır. Eğim, düzeydeki son değişim ile eski eğimin ağırlıklı ortalamasıdır. Tahmin, son düzeyden son eğimle uzatılan bir doğrudur.

> **Simge notu:** $`b_t`$: $`t`$ anındaki eğim (bir adımda beklenen artış) · $`\beta`$ *(beta)*: eğimin düzeltme katsayısı, 0 ile 1 arasında

**Not —** R'ın `ets()` çıktısındaki `beta`, 7.8.4'teki durum uzayı yazımına aittir ve buradaki $\beta$ ile $\alpha$'nın çarpımına eşittir; bu yüzden hep `alpha`'dan küçüktür.

Holt doğrusu sonsuza kadar aynı eğimle uzar. Uzun ufuklarda bu çoğu zaman fazla iyimserdir: Hiçbir büyüme sonsuza dek aynı hızla sürmez. Gardner ve McKenzie'nin (1985) önerdiği **sönümlü trend** (*damped trend*), eğimin tahmine katkısını her adımda bir $\phi$ oranıyla kısar:

```math
\hat{x}_{t+h|t} = \ell_t + \underbrace{(\phi + \phi^2 + \dots + \phi^h)}_{\substack{\text{her adımda azalan}\\\text{trend katkısı}}}\, b_t, \qquad 0 < \phi < 1
```

**Ne anlatıyor?** İlk adımda eğimin $\phi$ katı, ikinci adımda $\phi^2$ katı eklenir ve bu böyle sürer. Eklenen parçalar küçüldüğü için tahmin bir süre artar, sonra yataylaşır ve $\ell_t + \frac{\phi}{1 - \phi}\, b_t$ değerine yaklaşır. $\phi = 1$ Holt'un düz doğrusunu verir. (Düzey ve eğim denklemlerinde de $b_{t-1}$ yerine $\phi\, b_{t-1}$ yazılır.)

> **Simge notu:** $`\phi`$ *(fi)*: sönüm katsayısı (*damping parameter*); 7.1.1'deki AR katsayısıyla aynı harfle yazılır ama başka bir şeydir

Sayılarla: $\phi = 0.98$ ilk bakışta 1'e çok yakın görünür. Oysa 12 adımda eğimin toplam katkısı 12 değil, $0.98 + 0.98^2 + \dots + 0.98^{12} \approx 10.55$ eğimdir; 60 adımda 60 yerine yaklaşık 34.42, sonsuzda ise en fazla $0.98 / 0.02 = 49$ eğimdir. Küçük görünen bir sönüm, uzun ufukta trendin neredeyse yarısını siler. Bu sayılar 7.8.6'da önemli olacak.

#### 7.8.3. Holt-Winters: Mevsimsellik Eklemek

Winters (1960), Holt yöntemine üçüncü bir bileşen ekledi: **mevsim bileşeni** $s_t$. Bölüm 2.4'te gördüğümüz gibi mevsimsellik iki biçimde olabilir: **Toplamsal** (*additive*) modelde dalgaların boyu sabittir ve mevsim etkisi "ortalamanın 80 birim üstü" gibi bir farktır; **çarpımsal** (*multiplicative*) modelde dalgalar seviyeyle birlikte büyür ve etki "ortalamanın %20 üstü" gibi bir orandır. `AirPassengers` ikinci türdendir. 7.6'da bu büyümeyi log dönüşümüyle sabit boylu dalgalara çevirmiştik; Holt-Winters iki yol sunar: çarpımsal sürümle doğrudan orijinal seride çalışmak ya da log seride toplamsal sürümü kullanmak.

Çarpımsal Holt-Winters'ın denklemleri şunlardır (mevsim uzunluğunu, mevsim bileşeni $s_t$ ile karışmasın diye 7.3'teki $s$ yerine $m$ ile gösteriyoruz):

```math
\begin{aligned}
\ell_t &= \alpha\, \underbrace{\frac{x_t}{s_{t-m}}}_{\substack{\text{mevsimden arındırılmış}\\\text{gözlem}}} + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \\
b_t &= \beta\, (\ell_t - \ell_{t-1}) + (1 - \beta)\, b_{t-1} \\
s_t &= \gamma\, \underbrace{\frac{x_t}{\ell_{t-1} + b_{t-1}}}_{\substack{\text{bu dönemin}\\\text{mevsim oranı}}} + (1 - \gamma)\, \underbrace{s_{t-m}}_{\substack{\text{bir mevsim önceki}\\\text{aynı dönem}}} \\
\hat{x}_{t+h|t} &= (\ell_t + h\, b_t) \cdot \underbrace{s_{t+h-m}}_{\substack{\text{ilgili dönemin son}\\\text{mevsim çarpanı}}}
\end{aligned}
```

**Ne anlatıyor?** Düzey ve eğim Holt'taki gibi güncellenir, ama gözlem düzeye girmeden önce mevsim çarpanına bölünerek mevsimden arındırılır; böylece Temmuz'un yüksekliği düzeyin yükselmesi sanılmaz. Mevsim çarpanı da kendi SES'iyle güncellenir: bu dönemin gözlenen oranı ile geçen yılın aynı dönemindeki çarpanın ağırlıklı ortalaması. Tahmin, Holt doğrusunun ilgili dönemin çarpanıyla çarpılmasıdır. Son satır $h \le m$ içindir; daha uzak ufuklarda aynı dönemin en son çarpanı kullanılır.

> **Simge notu:** $`s_t`$: mevsim bileşeni (çarpımsal modelde oran, ör. 1.2 "%20 yukarı"; toplamsal modelde fark) · $`m`$: mevsim uzunluğu (aylık veride 12; `pmdarima`'daki `m` ile aynı, 7.7.4) · $`\gamma`$ *(gama)*: mevsim bileşeninin düzeltme katsayısı

Toplamsal sürümde bölmeler çıkarmaya, son satırdaki çarpım toplamaya dönüşür: $\ell_t = \alpha (x_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$, $s_t = \gamma (x_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma)\, s_{t-m}$ ve $\hat{x}_ {t+h|t} = \ell_t + h\, b_t + s_{t+h-m}$. Küçük bir örnek farkı gösterir: Son düzey 400, eğim 3 ve 7 ay sonraki Temmuz'un çarpanı 1.2 olsun. Çarpımsal tahmin $(400 + 7 \cdot 3) \cdot 1.2 = 505.2$'dir. Çarpan düzeyin %20'si demektir: Düzey 400 iken Temmuz etkisi yaklaşık 80, 800 iken 160 yolcudur. Toplamsal modelde Temmuz etkisi +80 ise tahmin $400 + 21 + 80 = 501$ olur ve etki düzey ne olursa olsun +80 kalır.

#### 7.8.4. ETS Çerçevesi: Hata, Trend, Mevsim

7.8.1–7.8.3'teki yöntemler uzun yıllar yalnızca **tahmin kuralları** olarak kullanıldı: Nokta tahmini üretiyorlardı ama arkalarında bir olasılık modeli yoktu; bu yüzden tahmin aralığı hesaplamak ya da iki yöntemi bilgi kriteriyle karşılaştırmak mümkün değildi. 1990'ların sonu ile 2000'lerin başında Ord, Koehler, Snyder ve Hyndman gibi araştırmacılar her yöntemi bir **durum uzayı modeli** (*state space model*) olarak yazdı ve aileye **ETS** adını verdi. Ad iki şekilde okunur: *ExponenTial Smoothing* ve modelin üç harfi olan *Error, Trend, Seasonal* (hata, trend, mevsim).

**Tanım:** Bir ETS modeli üç harfle adlandırılır: ETS(Hata, Trend, Mevsim). Hata A (*additive*, toplamsal) ya da M (*multiplicative*, çarpımsal); trend N (*none*, yok), A ya da Ad (*additive damped*, sönümlü toplamsal); mevsim N, A ya da M olabilir.

| Gösterim | Trend | Mevsim | Karşılık gelen yöntem |
| --- | --- | --- | --- |
| ETS(A,N,N) | yok | yok | Basit üstel düzeltme (7.8.1) |
| ETS(A,A,N) | toplamsal | yok | Holt (7.8.2) |
| ETS(A,Ad,N) | sönümlü | yok | Sönümlü Holt (7.8.2) |
| ETS(A,A,A) | toplamsal | toplamsal | Toplamsal Holt-Winters (7.8.3) |
| ETS(M,A,M) | toplamsal | çarpımsal | Çarpımsal Holt-Winters (7.8.3) |
| ETS(M,Ad,M) | sönümlü | çarpımsal | Sönümlü çarpımsal Holt-Winters |

Ortadaki ve sondaki harf, 7.8.1–7.8.3'te kurduğumuz nokta tahmin denklemlerini belirler. İlk harf (hata) ise yenidir ve rastgeleliğin seriye nasıl girdiğini söyler. En basit örnek ETS(A,N,N)'dir:

```math
\begin{aligned}
\underbrace{x_t}_{\text{gözlem}} &= \underbrace{\ell_{t-1}}_{\substack{\text{dünkü düzey}\\\text{(bir adım tahmin)}}} + \underbrace{\varepsilon_t}_{\text{şok}} \\
\underbrace{\ell_t}_{\text{yeni düzey}} &= \ell_{t-1} + \underbrace{\alpha\, \varepsilon_t}_{\substack{\text{şokun düzeye}\\\text{geçen payı}}}
\end{aligned}
```

**Ne anlatıyor?** İlk satır (gözlem denklemi), bugünkü değerin dünkü düzey artı bir şok olduğunu söyler; ikinci satır (durum denklemi), şokun $\alpha$ kadarının düzeye kalıcı olarak eklendiğini söyler. $\varepsilon_t = x_t - \ell_{t-1}$ ikinci satıra konursa 7.8.1'deki SES denklemi aynen çıkar: Nokta tahminleri değişmemiştir. Kazanılan şey, $\varepsilon_t$'nin artık bir olasılık dağılımına sahip olmasıdır ($\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$, normal dağılımlı).

> **Simge notu:** $`\varepsilon_t`$: bir adım ileri tahmin hatası; 7.1'deki beyaz gürültü şoku · **durum** (*state*): modelin her adımda güncellediği gizli bileşenler ($`\ell_t`$, $`b_t`$, $`s_t`$)

Olasılık modeli üç şey kazandırır:

1. **Olabilirlik:** Katsayılar ($\alpha$, $\beta$, $\gamma$, $\phi$ ve başlangıç durumları) ARIMA'daki gibi en çok olabilirlik yöntemiyle kestirilir (7.4).
2. **Model seçimi:** Her ETS modelinin bir AIC/AICc değeri vardır. R'daki `ets()` fonksiyonu, `auto.arima()` gibi uygun kombinasyonları dener ve AICc'si en küçük olanı seçer; toplamsal hatanın çarpımsal mevsimle birleştiği gibi sayısal olarak kararsız bazı kombinasyonları varsayılan olarak denemez. Python'da `statsmodels`'ın `ETSModel` sınıfı otomatik seçim yapmaz; adaylar tek tek kurulup `.aicc` değerleri karşılaştırılır.
3. **Tahmin aralıkları:** 7.6.7'deki gibi kuramsal tahmin aralıkları hesaplanabilir. Hata türü burada fark yaratır: Toplamsal hatada şokun büyüklüğü seviyeden bağımsızdır; çarpımsal hatada ($x_t = \ell_{t-1}(1 + \varepsilon_t)$) şok bir yüzdedir ve seri büyüdükçe aralıklar da genişler. `AirPassengers` gibi büyüyen serilerde bu daha gerçekçidir.

**ETS ile ARIMA'nın ilişkisi.** İki aile kısmen örtüşür. ETS(A,N,N)'de gözlem denklemini bir dönem öncesiyle farklayıp durum denklemini yerine koyarsak $x_t - x_{t-1} = \varepsilon_t + (\alpha - 1)\, \varepsilon_{t-1}$ buluruz: Bu, $\theta_1 = \alpha - 1$ olan bir ARIMA(0,1,1)'dir. Benzer şekilde Holt yöntemi ARIMA(0,2,2)'ye, sönümlü Holt ARIMA(1,1,2)'ye karşılık gelir; toplamsal Holt-Winters de özel bir SARIMA olarak yazılabilir. Ancak çarpımsal ETS modellerinin ARIMA karşılığı yoktur; tersine, AR terimli durağan ARIMA modellerinin de ETS karşılığı yoktur. Bu yüzden iki aile rakip değil, birbirini tamamlayan iki araç kutusudur.

#### 7.8.5. Theta Yöntemi

**Açıklama:** Theta yöntemi (Assimakopoulos ve Nikolopoulos, 2000), seriden eğrilikleri farklı ölçülerde büyütülmüş ya da küçültülmüş "çizgiler" türetir, her çizgiyi ona uygun basit bir yöntemle tahmin eder ve tahminleri birleştirir. Bu çizgilere **theta çizgileri** denir:

```math
\underbrace{Z_t(\theta)}_{\text{theta çizgisi}} = \theta\, x_t + (1 - \theta)\, \underbrace{(a + b\, t)}_{\substack{\text{seriye uydurulan}\\\text{regresyon doğrusu}}}
```

**Ne anlatıyor?** $\theta = 0$ iken çizgi, seriye uydurulan düz regresyon doğrusunun kendisidir: serinin uzun dönem eğilimi. $\theta = 1$ serinin kendisini verir. $\theta = 2$ iken serinin doğrudan sapmaları iki katına çıkar: Kısa dönemli iniş çıkışlar büyütülür. $Z_t(0)$ ile $Z_t(2)$'nin ortalaması tam olarak $x_t$'dir; yani bu iki çizgi seriyi "uzun dönem" ve "kısa dönem" parçalarına bölmenin bir yoludur.

> **Simge notu:** $`\theta`$ *(teta)*: burada theta çizgisinin eğrilik katsayısıdır; 7.1.2'deki MA katsayısıyla aynı harfle yazılır ama başka bir şeydir · $`a`$, $`b`$: regresyon doğrusunun kesişimi ve eğimi · $`n`$: son gözlemin sırası

Klasik Theta yöntemi bu iki çizgiyi kullanır:

```math
\hat{x}_{n+h|n} = \frac{1}{2}\, \underbrace{\big(a + b\, (n + h)\big)}_{\substack{\theta = 0\text{: doğru}\\\text{geleceğe uzatılır}}} + \frac{1}{2}\, \underbrace{\tilde{\ell}_n}_{\substack{\theta = 2\text{ çizgisine}\\\text{SES uygulanır}}}
```

**Ne anlatıyor?** Uzun dönem çizgisi düz bir doğru olarak geleceğe uzatılır, kısa dönem çizgisi 7.8.1'deki SES ile tahmin edilir ve iki tahminin ortalaması alınır. Mevsimsel serilerde önce seri mevsimden arındırılır (Bölüm 2.5'teki klasik çarpımsal ayrıştırmayla), yöntem arındırılmış seriye uygulanır, sonra mevsim çarpanları geri eklenir.

> **Simge notu:** $`\tilde{\ell}_n`$ *(el tilda)*: $`\theta = 2`$ çizgisine uygulanan SES'in son düzeyi

Bu kadar basit bir yöntem, 3003 serilik **M3 tahmin yarışmasında** (Makridakis ve Hibon, 2000) en başarılı yöntem olarak öne çıktı ve uzmanları şaşırttı. Hyndman ve Billah (2003) nedenini gösterdi: Klasik Theta tahmini, **sürüklenmeli** (*with drift*) bir SES'e eşittir ve sürüklenme, regresyon doğrusunun eğiminin yarısıdır ($b/2$). Yani Theta, trendi yarıya indirilmiş bir SES'tir: Gürültüye karşı SES kadar sakin, trende karşı temkinlidir. R'da `forecast::thetaf()`, Python'da `statsmodels.tsa.forecasting.theta.ThetaModel` bu yöntemi uygular; ikisi de $m$'inci gecikmedeki otokorelasyonu sınar ve mevsimsellik varsa seriyi kendiliğinden arındırır.

#### 7.8.6. Uygulama: `AirPassengers` ile ETS, Holt-Winters ve Theta

> **Uygulama dosyası:** [`Codes/R/ch07_sarima_airpassengers.R`](../Codes/R/ch07_sarima_airpassengers.R) (R kodları) ve [`Codes/python/ch07_sarima_airpassengers.py`](../Codes/python/ch07_sarima_airpassengers.py) · [Notebook](../Codes/notebooks/ch07_sarima_airpassengers.ipynb) · [![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch07_sarima_airpassengers.ipynb) (Python kodları)
>
> Bu alt bölümün kodları iki dosyanın sonundaki "7.8" başlıklı kısımlardadır; 7.8.1'deki elle hesap da R dosyasındadır.

Yöntemleri bu bölümdeki iki uygulamanın ayrımlarıyla deneyelim; böylece sonuçlar SARIMA ile doğrudan karşılaştırılabilir:

- **R:** Bölüm 8.3'teki ayrım: 1949–1959 eğitim (132 ay), 1960 test (12 ay). SARIMA olarak 7.6'daki havayolu modeli, log ölçekte.
- **Python:** 7.7.3'teki ayrım: 1949–1955 eğitim (84 ay), 1956–1960 test (60 ay). SARIMA olarak 7.7.5'teki model.

ETS, Holt-Winters ve Theta modellerini orijinal ölçekte kuruyoruz; büyüyen mevsim dalgasını çarpımsal mevsim bileşeni karşılar. Başarıyı üç metrikle ölçüyoruz: MAE (ortalama mutlak hata), RMSE ve MAPE (ortalama mutlak yüzde hata). Formülleri Bölüm 8.2'dedir; şimdilik üçü için de "küçük olan daha iyi" demek yeterlidir.

```r
library(forecast)
egitim <- window(AirPassengers, end = c(1959, 12))   # 1949-1959 (132 ay)
test   <- window(AirPassengers, start = c(1960, 1))  # 1960 (12 ay)
h <- length(test)

fit_ets <- ets(egitim)               # E, T, S harflerini AICc ile kendisi seçer
print(fit_ets)
#> ETS(M,Ad,M)
#>   Smoothing parameters:
#>     alpha = 0.758
#>     beta  = 0.0213
#>     gamma = 1e-04
#>     phi   = 0.98
#>   ...

fc_ets    <- forecast(fit_ets, h = h)
fc_hw     <- hw(egitim, seasonal = "multiplicative", h = h)   # Holt-Winters çarpımsal
fc_theta  <- thetaf(egitim, h = h)                            # Theta yöntemi
fc_sarima <- forecast(auto.arima(log(egitim)), h = h)         # havayolu modeli (log ölçek)
fc_snaive <- snaive(egitim, h = h)                            # mevsimsel naive

metrikler <- function(a, p) c(MAE  = mean(abs(a - p)),
                              RMSE = sqrt(mean((a - p)^2)),
                              MAPE = mean(abs((a - p) / a)) * 100)
a <- as.numeric(test)
tablo <- rbind(`Mevsimsel naive`        = metrikler(a, as.numeric(fc_snaive$mean)),
               `SARIMA (log)`           = metrikler(a, as.numeric(exp(fc_sarima$mean))),
               `ETS(M,Ad,M)`            = metrikler(a, as.numeric(fc_ets$mean)),
               `Holt-Winters çarpımsal` = metrikler(a, as.numeric(fc_hw$mean)),
               Theta                    = metrikler(a, as.numeric(fc_theta$mean)))
round(tablo, 2)
#>                          MAE  RMSE MAPE
#> Mevsimsel naive        47.83 50.71 9.99
#> SARIMA (log)           13.26 18.59 2.90
#> ETS(M,Ad,M)            22.80 27.40 4.66
#> Holt-Winters çarpımsal 11.13 16.58 2.37
#> Theta                  26.33 30.72 5.33
```

Kodu kısaca okuyalım:

1. `ets(egitim)`: Model türü verilmediği için uygun ETS kombinasyonlarını dener, AICc'ye göre birini seçer ve katsayılarını kestirir. `forecast(fit_ets, h = h)` 7.6.7'deki gibi nokta tahminlerini ve %80/%95 tahmin aralıklarını üretir.
2. `hw(egitim, seasonal = "multiplicative", h = h)`: Çarpımsal Holt-Winters. Arka planda ETS(M,A,M) kurar, ama katsayıları olabilirlik yerine bir adım ileri hataların karelerinin ortalamasını en küçük yapacak biçimde kestirir; bu yüzden sonucu `ets(egitim, model = "MAM")` ile birebir aynı değildir. Model kurmakla tahmin etmeyi tek adımda yapar ve doğrudan bir `forecast` nesnesi döndürür.
3. `thetaf(egitim, h = h)`: 7.8.5'teki Theta yöntemi; mevsimsellik testini ve arındırmayı kendisi yapar.
4. `auto.arima(log(egitim))` Bölüm 8.3'teki havayolu modelini yeniden kurar; tahminleri `exp()` ile yolcu sayısına çevrilir. `snaive()` her ayı 1959'un aynı ayıyla tahmin eder (8.1.2).
5. `metrikler()` ve `rbind()` Bölüm 8.3'teki tabloyu kurar (ayrıntılı açıklaması oradadır); ters tırnak, boşluk içeren satır adları içindir.

`ets()` **ETS(M,Ad,M)** modelini seçmiştir: çarpımsal hata, sönümlü trend, çarpımsal mevsim. `alpha = 0.758` düzeyin son gözlemleri yakından izlediğini, çok küçük `gamma` (0.0001) mevsim çarpanlarının neredeyse hiç güncellenmediğini, yani eğitimin başındaki mevsim deseninin sabit tutulduğunu, `phi = 0.98` ise trendin sönümlendiğini söyler (`beta` için 7.8.2'deki nota bakın).

Python tarafında aynı üç yöntemi `statsmodels` ile kuruyoruz (`train_data`, `test_data` ve `predictions_arima` 7.7'de tanımlandı):

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Holt-Winters
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # ETS (durum uzayı)
from statsmodels.tsa.forecasting.theta import ThetaModel  # Theta yöntemi

h = len(test_data)

# Holt-Winters: toplamsal trend, çarpımsal mevsimsellik
hw_model = ExponentialSmoothing(train_data, trend="add", seasonal="mul",
                                seasonal_periods=12).fit()
pred_hw = hw_model.forecast(h)

# ETS(M,Ad,M): çarpımsal hata, sönümlü trend, çarpımsal mevsimsellik
ets_model = ETSModel(train_data, error="mul", trend="add", damped_trend=True,
                     seasonal="mul", seasonal_periods=12).fit(disp=False)
pred_ets = ets_model.forecast(h)

# Theta: mevsimsellikten arındır, SES + sürüklenme uygula, mevsimselliği geri ekle
theta_model = ThetaModel(train_data, period=12).fit()
pred_theta = theta_model.forecast(h)
```

`ExponentialSmoothing` klasik Holt-Winters'tır ve katsayıları bir adım ileri hataların kareler toplamını en küçük yaparak kestirir. `ETSModel` 7.8.4'teki durum uzayı modelidir ve olabilirliği kullanır; otomatik seçim yapmadığı için model harflerini biz veriyoruz: R'ın `ets()` fonksiyonu bu 84 aylık eğitim setinde de ETS(M,Ad,M) seçer. `fit(disp=False)` optimizasyon mesajlarını gizler. `ThetaModel(train_data, period=12)` mevsim uzunluğunu alır, mevsimsellik testini ve arındırmayı kendisi yapar. Uygulama dosyası ayrıca mevsimsel naive tahmini (7.7.5'teki kutu) ve üç metriği hesaplayan küçük bir `metrikler()` fonksiyonu içerir.

**Sonuçlar.** İlk üç sayı sütunu R'daki 1960 testine, son üçü Python'daki 1956–1960 testine aittir (MAPE yüzde olarak; her sütunun en iyisi kalın):

| Model | 1960: MAE | RMSE | MAPE | 1956–60: MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| Mevsimsel naive | 47.83 | 50.71 | 9.99 | 112.43 | 126.01 | 27.05 |
| SARIMA (R: havayolu modeli, log; Python: 7.7.5) | 13.26 | 18.59 | 2.90 | 35.55 | 47.87 | 7.95 |
| ETS(M,Ad,M) | 22.80 | 27.40 | 4.66 | 40.80 | 55.34 | 9.14 |
| Holt-Winters çarpımsal | **11.13** | **16.58** | **2.37** | **27.68** | **37.12** | **6.22** |
| Theta | 26.33 | 30.72 | 5.33 | 58.86 | 72.98 | 13.53 |

Python dosyasındaki ikinci hücre, hataları (gerçek − tahmin) yıllara göre ortalar; artı değer, modelin gerçeğin altında kaldığını gösterir:

```text
      ARIMA   ETS  Holt-Winters  Theta
1956   17.5   8.2           8.8   16.3
1957   33.1  25.3          23.2   43.5
1958   19.7  19.7           9.9   43.2
1959   41.1  52.8          31.4   77.7
1960   62.9  89.4          53.4  112.6
```

**Çıktının yorumu:**

- **Çarpımsal Holt-Winters iki testte de en iyisidir.** 1960'ta fark küçüktür (RMSE 16.58'e karşı SARIMA 18.59) ve en büyük hatası yine Mart'tadır (−43.3; SARIMA'da −47.6, Bölüm 8.3'teki Paskalya açıklaması). 60 aylık testte fark belirgindir (37.12'ye karşı 47.87).
- **AICc'nin seçtiği model testte geride kalmıştır.** ETS(M,Ad,M) eğitim verisinde AICc'ye göre en iyi ETS modeliydi, ama iki testte de SARIMA'nın ve Holt-Winters'ın gerisindedir. 1960'taki hataları çoğunlukla artıdır ve en büyükleri yaz aylarındadır (Temmuz +52.1, Ağustos +38.4): Model özellikle yaz tepesini düşük tahmin etmiştir. İki neden vardır: $\gamma \approx 0$ mevsim desenini eğitimin başındaki hâliyle dondurmuştur; $\phi = 0.98$ ise trendi kısmıştır (7.8.2'deki hesapla 60 ayda trendin yaklaşık %43'ü kaybolur: $1 - 34.42/60 \approx 0.43$). Yıllık hata tablosunda bu açıkça görülür: ETS'nin ortalama hatası 1956'da 8.2 iken 1960'ta 89.4'e çıkar; Holt-Winters'ta 8.8'den 53.4'e. Ders, 7.7.5'teki alıştırmanınkiyle aynıdır: Bilgi kriteri eğitim verisine uyumu ölçer, test başarısını garanti etmez.
- **Theta bu seride en zayıf modeldir** (yine de mevsimsel naive'in çok önündedir). R'da sürüklenme ayda yaklaşık 1.28 bin yolcudur (`fc_theta$model$drift`); oysa 1958 ve 1959 yıllık ortalamaları (381.0 ve 428.3) arasındaki fark ayda yaklaşık 3.94 bin yolcuya karşılık gelir. "Eğimin yarısı" kuralı ve doğrusal büyüme varsayımı, yüzde olarak büyüyen bu seride Theta'yı geride bırakır. Theta'yı log seride kurmak (7.6'daki dönüşüm) sorunu büyük ölçüde giderir: 1960 testinde RMSE 30.72'den 21.48'e iner (R dosyasında bu satır da vardır).
- **60 ayda bütün modeller geride kalır.** Yıllık hata tablosunda dört modelin bütün yıllardaki ortalama hatası artıdır. 7.7.5'teki yorumda gördüğümüz gibi yıllık artış, yolcu cinsinden eğitim döneminde ortalama 26 dolayındayken test döneminde 38.4'tür; dört model de artışı yolcu cinsinden sabit (ya da sönümlü) bir eğimle sürdürdüğü için bu hızlanmayı eğitim verisinden göremez.
- **Tek bir test dönemi küçük bir örneklemdir.** 12 aylık bir testte iki model arasındaki yaklaşık 2 bin yolculuk RMSE farkı şans eseri olabilir. Güvenilir bir karşılaştırma için modeller birden çok test döneminde denenmelidir (Bölüm 16). Genel ders: Hiçbir aile her seride kazanmaz; ETS'yi ve SARIMA'yı birlikte denemek ucuzdur ve çoğu zaman öğreticidir.

> **Hemen deneyin 7.8.6 · Sönümlü mü, sönümsüz mü?** (R, ~4 dk)
>
> `ets()`'i sönümsüz trende zorlayıp aynı test yılında karşılaştıralım (yukarıdaki `egitim`, `test`, `h` ve `fit_ets` tanımlı olmalı):
>
> ```r
> fit_nd <- ets(egitim, damped = FALSE)                 # sönümsüz trende zorla
> print(fit_nd)
> c(sonumlu = fit_ets$aicc, sonumsuz = fit_nd$aicc)
> accuracy(forecast(fit_nd, h = h), test)[, c("MAE", "RMSE", "MAPE")]
> ```
>
> `damped = FALSE` yalnızca sönümsüz trendli modellerin denenmesini sağlar; diğer harfleri yine `ets()` seçer. `accuracy(tahmin, test)` hata metriklerini eğitim ve test satırlarıyla verir (8.3'teki not); model orijinal ölçekte kurulduğu için sonuçlar doğrudan yolcu birimindedir.
>
> **Ne göreceksiniz:** `ETS(M,A,M)`; `alpha = 0.4415`, `beta = 4e-04`, `gamma = 0.4696`. AICc 1262.941'dir, yani sönümlü modelin 1250.511'inden kötüdür. Ama `Test set` satırı `10.808561 16.45869 2.345369`'dur: MAE 10.81, RMSE 16.46, MAPE 2.35 ile tablodaki bütün modellerden iyidir. Bu model sönümü bırakmakla kalmamış, mevsim desenini de güncellemeyi seçmiştir ($`\gamma \approx 0.47`$).
>
> **Şimdi siz:** Aynı denemeyi log ölçekte yapın: `fit_log <- ets(log(egitim))` hangi modeli seçer? Tahminleri `exp()` ile geri çevirip `metrikler()` ile ölçün. Bu modelin AICc'si `fit_ets`'inkiyle karşılaştırılabilir mi?
>
> <details><summary>Cevap</summary>
>
> `ETS(A,A,A)`; `alpha = 0.7468`, `beta` ve `gamma` 1e-04. Log ölçekte çarpımsal mevsim toplamsala dönüşür (Bölüm 2.4) ve sönüm seçilmez. `round(metrikler(a, as.numeric(exp(forecast(fit_log, h = h)$mean))), 2)` → MAE 16.11, RMSE 23.49, MAPE 3.57: otomatik ETS(M,Ad,M)'den (RMSE 27.40) iyi, Holt-Winters'tan ve SARIMA'dan kötü. AICc'ler karşılaştırılamaz: −204.35 log seriye, 1250.51 orijinal seriye aittir; bilgi kriterleri yalnızca aynı veri üzerinde karşılaştırılır (7.4).
>
> </details>

**Kaynak:** Bu bölümdeki yöntemler için ücretsiz ve ayrıntılı bir başvuru kaynağı Hyndman, R. J. ve Athanasopoulos, G., *Forecasting: Principles and Practice*, 3. baskı, OTexts'tir (çevrim içi). Theta yöntemi: Assimakopoulos, V. ve Nikolopoulos, K. (2000), *International Journal of Forecasting*; Theta ile sürüklenmeli SES'in denkliği: Hyndman, R. J. ve Billah, B. (2003), aynı dergi.

---

Klasik modellerin gücü yorumlanabilirliklerindedir: Her katsayının açık bir anlamı vardır ve tahmin aralıkları kuramsal olarak hesaplanır. Ancak ARIMA ailesi doğrusal ilişkiler varsayar; karmaşık, doğrusal olmayan desenleri yakalamakta zorlanabilir. Bu tür durumlarda Bölüm 12'den itibaren ele alacağımız yapay zeka tabanlı yöntemler daha etkili olabilir. Ama önce, farklı modelleri adil biçimde karşılaştırabilmek için **model değerlendirme** araçlarını öğrenmemiz gerekiyor.

---

[İçindekiler](../Course_notes.md) · [← 6. Veri Manipülasyonu, Görselleştirme ve ACF/PACF](06_veri_manipulasyonu_gorsellestirme_ve_acf_pacf.md) · [8. Model Değerlendirme: Hata Metrikleri ve Eğitim-Test Ayrımı →](08_model_degerlendirme_hata_metrikleri_ve_egitim_test_ayrimi.md)
