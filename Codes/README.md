# Uygulama Kodları

Bu klasör, [ders notundaki](../Course_notes.md) uygulamaların çalıştırılabilir kodlarını içerir. Ders notunda kısa kodlar metnin içinde yer alır. Her bölümün kodlarının tamamı ise burada, baştan sona çalışan tek bir dosyada toplanmıştır. Ders notunda bir uygulamaya geldiğinizde, ilgili alt başlığın altındaki **💻 Uygulama dosyası** kutusu sizi buraya yönlendirir.

## Klasör yapısı

```
Codes/
├── R/               Bölüm 2–8: R betikleri (RStudio'da açın)
├── python/          Python betikleri (VS Code'da hücre hücre çalıştırılabilir)
├── notebooks/       Python betiklerinin Jupyter/Colab sürümleri
├── gretl/           Bölüm 11: Gretl betikleri
├── tools/py2nb.py   python/ → notebooks/ dönüştürücüsü
├── requirements.txt Python paketleri
└── install_packages.R  R paketleri
```

Veri dosyaları depo kökündeki [`data/`](../data) klasöründedir. Python kodları veriyi önce yerel `data/` klasöründe arar. Bulamazsa (örneğin Colab'da) GitHub'daki kopyasını otomatik olarak okur.

## Bölüm – dosya eşleşmesi

| Bölüm | Konu | R | Python | Notebook |
| --- | --- | --- | --- | --- |
| 2 | Bileşenler ve ayrıştırma | [ch02_ayristirma.R](R/ch02_ayristirma.R) | | |
| 3 | Durağanlık (ADF) | [ch03_duraganlik.R](R/ch03_duraganlik.R) | | |
| 4 | Tarih ve zaman nesneleri | [ch04_tarih_zaman.R](R/ch04_tarih_zaman.R) | | |
| 5 | `ts` ve `xts` | [ch05_ts_xts.R](R/ch05_ts_xts.R) | | |
| 6 | Manipülasyon, ACF/PACF | [ch06_manipulasyon_acf.R](R/ch06_manipulasyon_acf.R) | [ACF_PACF.py](ACF_PACF.py) (kısa örnek) | |
| 7 | ARIMA / SARIMA | [ch07_sarima_airpassengers.R](R/ch07_sarima_airpassengers.R) | [ch07_sarima_airpassengers.py](python/ch07_sarima_airpassengers.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch07_sarima_airpassengers.ipynb) |
| 8 | Model değerlendirme | [ch08_egitim_test.R](R/ch08_egitim_test.R) | [ch08_model_degerlendirme.py](python/ch08_model_degerlendirme.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch08_model_degerlendirme.ipynb) |
| 9 | Prophet | | [ch09_prophet.py](python/ch09_prophet.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch09_prophet.ipynb) |
| 10 | VAR | | [ch10_var.py](python/ch10_var.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch10_var.ipynb) |
| 11 | Gretl (ARIMA, VAR) | [ch11_arima_airpassengers.inp](gretl/ch11_arima_airpassengers.inp), [ch11_var.inp](gretl/ch11_var.inp) | | |
| 12 | Kayan pencere | | [ch12_kayan_pencere.py](python/ch12_kayan_pencere.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch12_kayan_pencere.ipynb) |
| 13 | XGBoost | | [ch13_xgboost.py](python/ch13_xgboost.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch13_xgboost.ipynb) |
| 15 | LSTM, GRU, 1D-CNN | | [ch15_derin_ogrenme.py](python/ch15_derin_ogrenme.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch15_derin_ogrenme.ipynb) |
| 16 | TimeSeriesSplit | | [ch16_timeseriessplit.py](python/ch16_timeseriessplit.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erkanozhan/AI_Based_Time_Series-Data_Analytics/blob/main/Codes/notebooks/ch16_timeseriessplit.ipynb) |

Bölüm 1, 14 ve 17'de kod yoktur. Bölüm 14'te Weka'nın grafik arayüzü kullanılır.

## Nasıl çalıştırılır?

Tüm komutlar **depo kök dizininden** çalıştırılmalıdır, yani `Course_notes.md` dosyasının bulunduğu klasörden.

### 1. Kurulum yapmadan: Google Colab

Tablodaki **Open in Colab** düğmesine tıklayın. Eksik paketler (ör. `pmdarima`) notebook'un ilk hücresinde kurulur. `Çalışma zamanı → Tümünü çalıştır` ile baştan sona çalıştırabilirsiniz.

### 2. Kendi bilgisayarınızda: Python

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r Codes/requirements.txt
python Codes/python/ch07_sarima_airpassengers.py
```

`Codes/python/` altındaki dosyalar **percent formatındadır**: `# %%` satırları dosyayı hücrelere böler. VS Code'da (Python ve Jupyter eklentileriyle) her hücrenin üstünde **Run Cell** bağlantısı görünür. Bu sayede kodu notebook gibi adım adım çalıştırıp ara sonuçları inceleyebilirsiniz.

### 3. R

RStudio'da depo klasörünü proje olarak açın (ya da çalışma dizinini depo kök dizini yapın) ve bir kez paketleri kurun:

```r
source("Codes/install_packages.R")
```

Sonra `Codes/R/` altındaki dosyayı açıp satır satır (`Ctrl+Enter`) çalıştırın. `# ---- 7.6.1 ... ----` biçimindeki satırlar RStudio'nun belge anahattında (outline) bölüm olarak görünür ve ders notundaki alt başlıklarla eşleşir. Komut satırından çalıştırmak için:

```bash
Rscript Codes/R/ch07_sarima_airpassengers.R
```

### 4. Gretl

Gretl'de `File → Script files → Open user file…` ile `Codes/gretl/` altındaki `.inp` dosyasını açın ve betik penceresindeki çalıştır düğmesine basın. Betikler veriyi `data/` klasöründen okur. Gretl'in çalışma dizini farklıysa betiğin başındaki açıklamaya bakın.

## Katkı / güncelleme (ders sorumlusu için)

Notebook'lar elle düzenlenmez; kaynak dosya `Codes/python/` altındaki `.py` dosyasıdır. Bir `.py` dosyasını değiştirdikten sonra notebook'u yeniden üretin:

```bash
python Codes/tools/py2nb.py Codes/python/ch13_xgboost.py
```

`# COLAB: pip install -q paket` satırları `.py` dosyasında yorum satırıdır. Notebook'ta ise `%pip install` hücresine dönüşür.
