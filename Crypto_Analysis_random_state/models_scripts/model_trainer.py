from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os
import joblib

# Gradient Boosting Regression ile kripto para fiyat tahmini yapan modül.
def train_model(symbol, interval):
    # Belirli bir kripto para için model eğitimi yapar.
    try:
        # Veri yolu
        data_path = f"d:/Crypto_Analysis_random_state/data/processed/{interval}_processed/{symbol}.xlsx"
        
        # Veri okuma
        df = pd.read_excel(data_path)
        
        # Özellik seçimi
        target = df['close']
        features = df.drop(['timestamp', 'close'], axis=1)
        
        # Veri bölme
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Model ayarları
        if interval.lower() == 'monthly':
            params = {
                'n_estimators': 300,    # Ağaç sayısı
                'learning_rate': 0.03,  # Öğrenme hızı
                'max_depth': 4,         # Maksimum Ağaç Derinliği :Daha büyük derinlik Daha fazla (overfitting) riski girer
                'min_samples_split': 4, # Minimum Bölme Sayısı :Büyük değerler → Model daha az bölünür → Daha basit model → Overfitting azalır.
                'min_samples_leaf': 3,  # Yaprak Düğümdeki Minimum Örnek Sayısı :Daha yüksek değer → Model daha basitleşir → Overfitting azalır.
                'subsample': 0.8,       #Alt Örnekleme Oranı :subsample = 1.0 → Tüm veri setini kullanır (overfitting riski daha yüksek).1 kucukse of daha az
                'max_features': 0.7     # Maksimum Özellik Sayısı :Daha küçük değer → Model her iterasyonda farklı özellikleri kullanır → Daha genel model oluşur.
            }
        elif interval.lower() == 'hourly':
            params = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'subsample': 0.8,
                'max_features': 0.8
            }
        else:  # daily
            params = {
                'n_estimators': 400,
                'learning_rate': 0.02,
                'max_depth': 5,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'subsample': 0.85,
                'max_features': 0.75
            }
        
        # Model oluşturma
        model = GradientBoostingRegressor(**params, random_state=42, verbose=1)
        
        # Eğitim
        print(f"\n{symbol} ({interval}) modeli eğitiliyor...")
        model.fit(X_train, y_train)
        
        # Tahmin
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Performans ölçümü
        r2_train = r2_score(y_train, train_pred)
        r2_test = r2_score(y_test, test_pred)
        mse_train = mean_squared_error(y_train, train_pred)
        mse_test = mean_squared_error(y_test, test_pred)
        
        # MAE ve RMSE hesapla (test verisi için)
        mae_test = mean_absolute_error(y_test, test_pred)
        rmse_test = np.sqrt(mse_test)
        
        # Özellik önemi
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Model kayıt
        model_dir = f"d:/Crypto_Analysis_random_state/models/gradient_boosting/{interval.lower()}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Model dosyası
        model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
        joblib.dump(model, model_path)
        
        # Sonuçları kaydet
        performance = {
            'symbol': symbol,
            'interval': interval,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'features': list(features.columns),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        performance_path = os.path.join(model_dir, f"{symbol}_performance.joblib")
        joblib.dump(performance, performance_path)
        
        # Sonuçları göster
        print(f"Model kaydedildi: {model_path}")
        print(f"Performans metrikleri kaydedildi: {performance_path}")
        print(f"R² (Eğitim): {r2_train:.4f}")
        print(f"R² (Test): {r2_test:.4f}")
        print(f"MSE (Eğitim): {mse_train:.4f}")
        print(f"MSE (Test): {mse_test:.4f}")
        print(f"MAE (Test): {mae_test:.4f}")
        print(f"RMSE (Test): {rmse_test:.4f}")
        print("\nÖnemli özellikler:")
        print(feature_importance.head())
        
        return performance
        
    except Exception as e:
        print(f"Hata: {symbol} ({interval}) eğitilirken bir hata oluştu:")
        print(str(e))
        return None

def train_all_models():
    # Tüm kripto paralar için model eğitimi yapar.
    # Kripto listesi
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK', 
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    
    # Zaman aralıkları
    intervals = ['Daily', 'Monthly', 'Hourly']
    
    # Eğitim başlat
    for interval in intervals:
        print(f"\n{interval} modelleri eğitiliyor...")
        for symbol in symbols:
            try:
                train_model(symbol, interval)
            except Exception as e:
                print(f"Hata: {symbol} ({interval}) eğitilirken bir hata oluştu:")
                print(str(e))

if __name__ == "__main__":
    train_all_models()
