# Yapay Zeka Tabanlı Zaman Serisi ve Veri Analizi

Bu depo, zaman serisi analizini klasik istatistiksel yöntemlerden yapay zeka tabanlı modellere kadar adım adım ele alan bir üniversite dersinin ders notlarını, uygulama kodlarını ve veri setlerini içerir. Ders; zaman serisinin temel kavramlarını ve durağanlığı, R'da tarih ve zaman serisi nesnelerini, ACF/PACF ile model belirlemeyi, ARIMA/SARIMA, Prophet ve VAR modellerini, model değerlendirme yöntemlerini ve XGBoost, LSTM, GRU, 1D-CNN gibi makine öğrenmesi ve derin öğrenme yaklaşımlarını kapsar. Uygulamalarda R, Python, Gretl ve Weka kullanılır.

## Ders notları

Bölümlerin listesi ve bağlantıları için **[İçindekiler (Course_notes.md)](Course_notes.md)** sayfasına bakılabilir. Her bölüm `notlar/` klasöründe ayrı bir dosyadır. Ekler:

- [Ek A: R'a Giriş](R_Giris.md)
- [Ek B: Gretl'e Giriş](Gretl.md)

## Klasör yapısı

| Klasör / dosya | İçerik |
| --- | --- |
| [`Course_notes.md`](Course_notes.md) | Ders notlarının içindekiler sayfası |
| [`notlar/`](notlar) | Bölüm 1–17'nin ders notları |
| [`Codes/`](Codes/README.md) | Çalıştırılabilir R betikleri, Python betikleri, Jupyter/Colab notebook'ları ve Gretl betikleri |
| [`data/`](data) | Uygulamalarda kullanılan veri setleri (`AirPassengers.csv`, `macro.csv`) |
| [`images/`](images) | Ders notlarındaki şekiller |
| `R_Giris.md`, `Gretl.md` | Ek A ve Ek B |

## Kurulum ve çalıştırma

Bütün komutlar depo kök dizininden çalıştırılır. Ayrıntılı açıklama [`Codes/README.md`](Codes/README.md) dosyasındadır.

**R:** R ve RStudio kurulduktan sonra gerekli paketler bir kez kurulur:

```r
source("Codes/install_packages.R")
```

**Python:** Bir sanal ortam oluşturulup paketler kurulur:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r Codes/requirements.txt
```

Kurulum yapmadan çalışmak için `Codes/README.md` tablosundaki Colab bağlantıları kullanılabilir.

**Gretl:** [gretl.sourceforge.net](https://gretl.sourceforge.net) adresinden kurulur. Derste kullanılan örnek veri dosyaları (`bjg`, `djclose`, `nile`, `sunspots`) Gretl ile birlikte gelir; betikler `Codes/gretl/` klasöründedir.

## Yazım kuralları

- Ondalık ayırıcı olarak nokta kullanılır (R ve Python çıktılarıyla uyum için). Türkçe arayüzlü Excel ya da LibreOffice Calc çıktıları anlatılırken ekranda görünen virgüllü yazım korunur.
- Terimlerin İngilizce karşılıkları ilk geçtikleri yerde parantez içinde verilir.

## Lisans

Bu depo [MIT Lisansı](LICENSE) ile lisanslanmıştır.
