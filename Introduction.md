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
            A[Veri Toplama ve Görselleştirme] --> B{"Seri Durağan mı?"};
            B -- "Hayır" --> C[Fark Alma (Differencing)];
            C --> D[ACF/PACF ile Model Belirleme (p,q)];
            B -- "Evet" --> D;
            D --> E[Model Parametrelerini Tahmin Etme];
            E --> F{"Model Tanısal Kontrolleri (Artıklar Beyaz Gürültü mü?)"};
            F -- "Evet" --> G[Tahmin (Forecasting)];
            F -- "Hayır" --> D;
            subgraph "Box-Jenkins Metodolojisi"
                direction LR
                D -- "p,d,q" --> E -- "Model" --> F
            end
        ```

#### Zaman Serisi Analizinde Görselleştirme

Görselleştirme, veri içindeki desenleri, anormallikleri, trendleri ve mevsimselliği insan gözüyle hızlıca tespit etmenin en etkili yoludur.

-   **Zaman Serisi Grafiği:** Verinin zamana karşı çizilmesi. Örneğin, bir şehrin saatlik elektrik tüketim grafiği, gün içindeki (sabah ve akşam) ve hafta sonundaki talep düşüşlerini net bir şekilde gösterir.
-   **ACF ve PACF Grafikleri:** Model belirleme aşamasında kritik öneme sahiptir. Örneğin, bir ACF grafiğinde yavaşça azalan çubuklar trendin varlığına işaret ederken, her 12 ayda bir tekrar eden belirgin bir çubuk yıllık mevsimselliği gösterir.
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
