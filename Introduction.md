**Yapay Zeka Tabanlı Zaman Serisi ve Veri Analiği**

---

### Zaman Serisi Analizi: Giriş

Zaman serisi, belirli bir zaman aralığında ardışık olarak gözlemlenen veri noktalarının bütünüdür. Örneğin, bir hastanedeki günlük hasta sayısı, bir şirketin aylık satış rakamları veya bir meteoroloji istasyonunda saatlik sıcaklık ölçümleri zaman serisi verilerine örnektir. Box ve Jenkins’e göre zaman serisi, “zamana bağlı olarak düzenli aralıklarla kaydedilen gözlemler dizisidir.” Zaman serisi analizi, geçmiş verilerden öğrenerek gelecekteki değerleri tahmin etmeye, anormallikleri tespit etmeye ve temel desenleri ortaya çıkarmaya olanak tanır.

Zaman serisi analizi finans (hisse senedi fiyatları, döviz kurları), ekonomi (enflasyon, işsizlik oranları), sağlık (hasta takibi, salgın analizi), enerji (elektrik tüketimi, üretim tahmini), mühendislik (sensör verileri, arıza tahmini) ve çevre bilimleri (hava durumu, iklim değişikliği) gibi birçok alanda kritik öneme sahiptir. Örneğin, elektrik talebinin saatlik değişimi enerji yönetiminde, kalp atış hızının zaman içindeki değişimi ise sağlıkta önemli bilgiler sunar.
Zaman serisi verilerinin analizi, aşağıdaki temel soruları yanıtlamayı amaçlar:
- Geçmişteki desenler nelerdir?
- Gelecekteki değerler nasıl tahmin edilebilir?
- Serideki olağan dışı değişimler nasıl tespit edilir?
- Zaman serisinin bileşenleri (trend, mevsimsellik, döngü, rastgelelik) nelerdir?

Bu nedenle zaman serisi analizi, veri bilimi ve yapay zeka uygulamalarında önemli bir rol oynar ve karar destek sistemlerinin temelini oluşturur.

#### Zaman Serisi Temel Kavramları

- **Gözlem (Observation):** $x_t$ ile gösterilir, $t$ zamanındaki veri noktasıdır.
    - *Örnek:* Bir bankadaki günlük işlem sayısı, $x_{15}$ = 120 işlem.
    - **Şekil:**
        **Gözlem (Observation):** $x_t$ ile gösterilir, $t$ zamanındaki veri noktasıdır.
        - *Örnek:* Bir bankadaki günlük işlem sayısı, $x_{15}$ = 120 işlem.
        - **Şekil:**

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

        ---

        - **Zaman Dizisi (Time Index):** $t = 1, 2, ..., T$ şeklinde tanımlanır.
            - *Örnek:* Bir yıl boyunca her ay ölçülen sıcaklık değerleri için $t = 1$ Ocak, $t = 12$ Aralık.
            - **Şekil:**

        ```markdown
        Zaman Dizisi:
        t:   1   2   3   ...  T
                 │   │   │        │
                 ●───●───●───...──●
        ```

        ---

        - **Trend:** Zaman serisindeki uzun vadeli artış veya azalış.
            - *Örnek:* Bir e-ticaret sitesinin yıllık satışlarının sürekli artması.
            - **Şekil:**

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

        ---

        - **Mevsimsellik (Seasonality):** Belirli periyotlarda tekrar eden dalgalanmalar.
            - *Örnek:* Yaz aylarında artan dondurma satışları, kışın azalan satışlar.
            - **Şekil:**

        ```markdown
        Mevsimsellik:
        Değer
            5 |   ●   ●   ●
            3 | ●   ●   ●
            1 |●___●___●___
                 t →
        ```

        ---

        - **Durağanlık (Stationarity):** Serinin istatistiksel özelliklerinin zamanla değişmemesi.
            - *Örnek:* Ortalama ve varyansı yıl boyunca sabit kalan sıcaklık ölçümleri.
            - **Şekil:**

        ```markdown
        Durağanlık:
        Değer
            5 | ● ● ● ● ● ●
            3 | ● ● ● ● ● ●
            1 |●_●_●_●_●_●_
                 t →
        ```

        Bu kavramlar, zaman serisi analizinin temelini oluşturur. Her bir şekil, ilgili kavramı görsel olarak daha anlaşılır ve sade biçimde temsil eder.

#### Zaman Serisi Bileşenleri

Bir zaman serisi genellikle şu şekilde modellenir:
$$
x_t = T_t + S_t + C_t + I_t
$$
- $T_t$: Trend bileşeni
- $S_t$: Mevsimsellik bileşeni
- $C_t$: Döngüsel bileşen
- $I_t$: Rastgele (iradi) bileşen
#### Zaman Serisi Sınıflandırması ve Tipleri

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

#### Zaman Serisi Analizinde Kullanılan Yöntemler ve Modeller

Zaman serisi analizi, geçmiş verilerdeki kalıpları anlayarak geleceği öngörmeyi amaçlayan istatistiksel ve makine öğrenmesi yöntemlerini kapsar. Yöntem seçimi, verinin yapısına ve analizin amacına bağlıdır.

1.  **Keşifsel Analiz ve Temel İstatistikler**
    *   **Otokorelasyon Fonksiyonu (ACF):** Bir zaman serisinin, kendi geçmiş (gecikmeli) değerleriyle ne kadar ilişkili olduğunu ölçer. Örneğin, dondurma satışlarının bugünkü değeri, dünkü değeriyle yüksek korelasyon gösterebilir (lag-1). Aynı zamanda, 7 gün önceki değeriyle de (haftalık mevsimsellik nedeniyle) yüksek korelasyon gösterebilir (lag-7). ACF grafiği, bu tür periyodik desenleri ve mevsimselliği tespit etmek için kritik bir araçtır.
        *   **Formül:**
            $$
            \rho_k = \frac{\text{Cov}(x_t, x_{t-k})}{\text{Var}(x_t)} = \frac{\sum_{t=k+1}^{T} (x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{T} (x_t - \bar{x})^2}
            $$
    *   **Kısmi Otokorelasyon Fonksiyonu (PACF):** İki zaman noktası arasındaki doğrudan ilişkiyi, aralarındaki gecikmelerin etkisini ortadan kaldırarak ölçer. Örneğin, $x_t$ ve $x_{t-2}$ arasındaki PACF, $x_{t-1}$'in aracı etkisini kaldırarak bu iki nokta arasındaki net korelasyonu gösterir. ACF ve PACF, Box-Jenkins metodolojisinde ARIMA modellerinin mertebelerini (p, q) belirlemek için birlikte kullanılır (Box & Jenkins, 1970).

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
        ```

#### Zaman Serisi Analizinde Görselleştirme

Görselleştirme, veri içindeki desenleri, anormallikleri, trendleri ve mevsimselliği insan gözüyle hızlıca tespit etmenin en etkili yoludur.

-   **Zaman Serisi Grafiği:** Verinin zamana karşı çizilmesi. Örneğin, bir şehrin saatlik elektrik tüketim grafiği, gün içindeki (sabah ve akşam) ve hafta sonundaki talep düşüşlerini net bir şekilde gösterir.

    **Python ile Airlines Veri Seti Görselleştirme Örneği:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Airlines veri setini yükle (örnek: 'AirPassengers.csv')
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
    
**Zaman Serisi Formatına Dönüştürülmüş Veri Örneği**

Zaman serisi analizi için her veri başlangıçta uygun formatta olmayabilir. Örneğin, bir e-ticaret sitesinin ham işlem kayıtları (transaction logs) genellikle "işlem zamanı", "müşteri ID", "ürün" gibi sütunlar içerir ve zaman serisi formatında değildir. Ancak bu veriler, belirli bir zaman aralığına (örneğin günlük) göre gruplanarak zaman serisi haline getirilebilir.

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

Bu örnekte, ham işlem kayıtları günlük bazda gruplanarak zaman serisi formatına dönüştürülmüş ve görselleştirilmiştir. Böylece veri, zaman serisi analizine uygun hale getirilmiştir.

-   **ACF (Autocorrelation Function - Otokorelasyon Fonksiyonu) ve PACF (Partial Autocorrelation Function - Kısmi Otokorelasyon Fonksiyonu) Grafikleri:** Model belirleme aşamasında kritik öneme sahiptir.  
    - **ACF Notasyonu:** $\rho_k$ ile gösterilir ve bir zaman serisinin kendi gecikmeli (lagged) değerleriyle olan korelasyonunu ölçer.  
    - **PACF Notasyonu:** $\phi_{kk}$ ile gösterilir ve iki zaman noktası arasındaki doğrudan ilişkiyi, aradaki diğer gecikmelerin etkisini ortadan kaldırarak ölçer.  

    **Örnek Hesaplama:**
    Diyelim ki elimizde 5 günlük sıcaklık verisi var: $x = [20, 22, 21, 23, 24]$

    - **ACF (Lag-1) Hesabı:**
        1. Ortalama: $\bar{x} = (20 + 22 + 21 + 23 + 24)/5 = 22$
        2. $\rho_1 = \frac{\sum_{t=2}^{5} (x_t - \bar{x})(x_{t-1} - \bar{x})}{\sum_{t=1}^{5} (x_t - \bar{x})^2}$
        3. Hesaplanan değerler:
            - $(22-22)(20-22) = 0$
            - $(21-22)(22-22) = -1 \times 0 = 0$
            - $(23-22)(21-22) = 1 \times -1 = -1$
            - $(24-22)(23-22) = 2 \times 1 = 2$
            - Toplam: $0 + 0 + (-1) + 2 = 1$
        4. Payda: $(20-22)^2 + (22-22)^2 + (21-22)^2 + (23-22)^2 + (24-22)^2 = 4 + 0 + 1 + 1 + 4 = 10$
        5. $\rho_1 = 1 / 10 = 0.1$

    - **PACF Hesabı:** PACF, örneğin lag-2 için, $x_t$ ile $x_{t-2}$ arasındaki doğrudan ilişkiyi, $x_{t-1}$'in etkisini çıkararak ölçer. Küçük veri setlerinde genellikle istatistiksel paketler ile hesaplanır.

        **Örnek Hesaplama:**
        Diyelim ki elimizde 5 günlük sıcaklık verisi var: $x = [20, 22, 21, 23, 24]$

        - **PACF (Lag-2) Hesabı:**
            1. $x_t$ ile $x_{t-2}$ arasındaki korelasyonu, $x_{t-1}$'in etkisini ortadan kaldırarak bulmak gerekir.
            2. Bu işlem, genellikle regresyon ile yapılır: $x_t = a_1 x_{t-1} + a_2 x_{t-2} + \epsilon_t$
            3. $a_2$ katsayısı, lag-2 PACF değerini verir.
            4. Küçük veri setlerinde elle hesaplamak zordur, ancak istatistiksel paketler (örneğin Python'da `statsmodels` veya R'da `pacf` fonksiyonu) ile kolayca bulunabilir.

        **Python ile PACF Hesabı Örneği:**
        ```python
        import numpy as np
        from statsmodels.tsa.stattools import pacf

        x = np.array([20, 22, 21, 23, 24])
        pacf_values = pacf(x, nlags=2)
        print("Lag-2 PACF:", pacf_values[2])
        ```

        **R ile PACF Hesabı Örneği:**
        ```r
        x <- c(20, 22, 21, 23, 24)
        pacf_result <- pacf(x, plot = FALSE)
        cat("Lag-2 PACF:", pacf_result$acf[2], "\n")
        ```

        **PACF Kavramsal Gösterimi:**

        ```mermaid
            graph TD
                subgraph "PACF Hesabı"
                X_t_2["x_{t-2}"] -->|Doğrudan Etki| X_t["x_t"]
                X_t_1["x_{t-1}"] -.->|Dolaylı Etki| X_t
                end
                X_t_2 -.->|Dolaylı Etki| X_t_1
        ```

        **Açıklama:**  
        PACF (Kısmi Otokorelasyon Fonksiyonu), bir zaman serisinde iki nokta arasındaki doğrudan ilişkiyi ölçer ve aradaki diğer gecikmelerin etkisini ortadan kaldırır. Yukarıdaki şekilde, $x_{t-2}$'nin $x_t$ üzerindeki doğrudan etkisi (solid arrow) ve dolaylı etkisi ($x_{t-1}$ üzerinden, dashed arrow) gösterilmiştir. PACF, sadece doğrudan etkiyi ölçer.

        **Hesaplama Örneği:**  
        Diyelim ki elimizde 5 günlük sıcaklık verisi var: $x = [20, 22, 21, 23, 24]$

        - PACF (Lag-2) değerini bulmak için $x_t$'yi hem $x_{t-1}$ hem de $x_{t-2}$'ye karşı regresyon ile modelleyin:  
          $x_t = a_1 x_{t-1} + a_2 x_{t-2} + \epsilon_t$  
          Burada $a_2$ katsayısı, lag-2 PACF değerini verir.

        **Python ile PACF Hesabı:**
        ```python
        import numpy as np
        from statsmodels.tsa.stattools import pacf

        x = np.array([20, 22, 21, 23, 24])
        pacf_values = pacf(x, nlags=2)
        print("Lag-2 PACF:", pacf_values[2])
        ```

        **R ile PACF Hesabı:**
        ```r
        x <- c(20, 22, 21, 23, 24)
        pacf_result <- pacf(x, plot = FALSE)
        cat("Lag-2 PACF:", pacf_result$acf[2], "\n")
        ```

        - **Yorum:** PACF grafiğinde, belirli bir gecikme için çubuğun yüksek olması, o gecikmenin seride doğrudan etkili olduğunu gösterir.

    - **Yorum:** ACF grafiğinde çubukların yavaşça azalması trendin varlığına, belirli aralıklarda tekrar eden yüksek çubuklar ise mevsimselliğe işaret eder.


-   **Mevsimsel Ayrıştırma Grafiği (Seasonal Decomposition Plot):** Seriyi bileşenlerine ayırarak görselleştirir: Trend, Mevsimsellik ve Artıklar (Rastgele Gürültü). Bu, bir e-ticaret sitesinin satış verilerinde uzun vadeli büyüme trendini, tatil dönemlerindeki mevsimsel artışlardan ve beklenmedik satış dalgalanmalarından ayırmayı sağlar.

    ```mermaid
    graph TD
        subgraph "Zaman Serisi Ayrıştırması"
            A(Orijinal Seri: x_t) --> B(Trend: T_t);
            A --> C(Mevsimsellik: S_t);
            A --> D(Artıklar/Gürültü: I_t);
        end
    ```

#### Yapay Zeka ile Zaman Serisi Analizi

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

Bu bölümde, zaman serisi analizinde kullanılan temel istatistiksel yöntemlerden modern yapay zeka yaklaşımlarına kadar geniş bir yelpaze sunulmuştur. Her yöntemin kendine özgü avantajları olup, doğru modelin seçimi probleme ve verinin doğasına bağlıdır.

**Referanslar:**
-   Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
-   Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.
-   Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735-1780.
