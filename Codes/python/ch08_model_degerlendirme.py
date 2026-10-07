# %% [markdown]
# # Bölüm 8 — Model Değerlendirme: `evaluate_model()` ile SARIMA ve Mevsimsel Naive
# Ders notu: Course_notes.md, Bölüm 8.4. Bu dosya bölümdeki Python kodlarının tamamını içerir.
# Çalıştırma: depo kök dizininden `python Codes/python/ch08_model_degerlendirme.py`
# veya VS Code'da hücre hücre (# %%) ya da Colab'da notebook sürümü.
#
# Dosya kendi başına çalışır: Bölüm 7.7'deki veri hazırlığı ve SARIMA modeli aşağıda kısaca yeniden kurulur.

# %%
# COLAB: pip install -q pmdarima
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima.datasets import load_airpassengers
from pmdarima import auto_arima

# %% [markdown]
# ## Hazırlık: Bölüm 7.7'deki veri, eğitim-test ayrımı ve SARIMA tahminleri
# Son 60 ay (1956-1960) test seti; model `auto_arima` ile yalnızca eğitim verisinden seçilir.

# %%
data = load_airpassengers(as_series=True)
data.index = pd.date_range(start="1949-01-01", periods=len(data), freq="MS")

train_data = data[:-60]
test_data = data[-60:]

auto_model = auto_arima(train_data,
                        seasonal=True,
                        m=12,
                        stepwise=True,
                        suppress_warnings=True,
                        trace=False)
print("Seçilen model:", auto_model)

predictions_arima = auto_model.predict(n_periods=len(test_data))
predictions_arima = pd.Series(np.asarray(predictions_arima), index=test_data.index)

# %% [markdown]
# ## 8.4.1 `evaluate_model()` fonksiyonu

# %%
def evaluate_model(y_true, y_pred, model_name):
    """
    Model performansını değerlendirir ve sonuçları yazdırır.

    Parametreler:
        y_true: Gerçek değerler (array veya Series)
        y_pred: Tahmin edilen değerler (array veya Series)
        model_name: Modelin adı (string)

    Döndürür:
        dict: MAE, RMSE ve MAPE değerlerini içeren sözlük
    """
    # Array'e dönüştür (Series indeksleri farklı olsa bile sıraya göre karşılaştırılır)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Metrikleri hesapla
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE hesaplarken sıfıra bölmeyi önle (gerçek değeri 0 olan noktalar hesaba katılmaz)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Sonuçları yazdır
    print(f"\n{'=' * 40}")
    print(f"{model_name} Performans Sonuçları")
    print(f"{'=' * 40}")
    print(f"MAE:  {mae:>10.2f}")
    print(f"RMSE: {rmse:>10.2f}")
    print(f"MAPE: {mape:>9.2f}%")

    return {'mae': mae, 'rmse': rmse, 'mape': mape}

# %% [markdown]
# ## 8.4.2 Örnek kullanım
# SARIMA tahminlerini 8.1.2'deki mevsimsel naive referansla karşılaştırıyoruz.

# %%
# 1) Bölüm 7.7'deki SARIMA (auto_arima) tahminleri
arima_metrics = evaluate_model(test_data, predictions_arima, "SARIMA (auto_arima)")

# 2) Referans model: mevsimsel naive
#    Eğitim setinin son 12 ayı, 60 aylık test dönemi boyunca tekrar edilir.
son_yil = train_data[-12:].values
snaive_pred = np.tile(son_yil, len(test_data) // 12 + 1)[:len(test_data)]
snaive_metrics = evaluate_model(test_data, snaive_pred, "Mevsimsel Naive")

# 3) Sonraki bölümlerde aynı fonksiyonu diğer modeller için de kullanacağız.
#    (Bu satırlar, ilgili bölümlerdeki değişkenler tanımlandıktan sonra çalışır.)
#    Dikkat: Prophet ve XGBoost bölümlerinde test seti yalnızca 1960 yılıdır (12 ay);
#    onları 8.3'teki 12 aylık sonuçlarla kıyaslayın. LSTM ise buradaki 60 aylık
#    test dönemini (1956-1960) kullanır.
# prophet_metrics = evaluate_model(test['y'], tahmin, "Prophet")          # Bölüm 9.3 (12 ay)
# xgb_metrics     = evaluate_model(y_test, y_test_pred, "XGBoost")        # Bölüm 13.4 (12 ay)
# lstm_metrics    = evaluate_model(testY_inv, test_predict[:, 0], "LSTM") # Bölüm 15 (60 ay)

# 4) Sonuçları tek bir tabloda toplayalım
sonuclar = pd.DataFrame({"SARIMA": arima_metrics,
                         "Mevsimsel Naive": snaive_metrics}).T
print(sonuclar.round(2))
