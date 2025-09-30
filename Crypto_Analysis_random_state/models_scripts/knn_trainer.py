import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def train_model(symbol, interval):
    #KNN Regressor modelini eğitir ve performans metriklerini hesaplar.
   
    try:
        base_dir = "d:/Crypto_Analysis_random_state"
        data_dir = os.path.join(base_dir, "data", "processed", f"{interval}_processed")
        model_dir = os.path.join(base_dir, "models", "knn", interval)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Veriyi yükle
        data_file = os.path.join(data_dir, f"{symbol}.xlsx")
        if not os.path.exists(data_file):
            print(f"{symbol} için veri bulunamadı: {data_file}")
            return None
            
        df = pd.read_excel(data_file)
        
        # Hedef ve özellikler
        target = df['close']
        features = df.drop(['timestamp', 'close'], axis=1)
        
        # Veri bölme
        if interval.lower() == 'monthly':
            # Aylık: %20 test
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.20, random_state=42
            )
        else:
            # Diğer: %25 test
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.25, random_state=42
            )
            
        # Ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model parametreleri
        if interval.lower() == 'monthly':
            params = {
                'n_neighbors': 8,  # Arttırıldı
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 30
            }
        elif interval.lower() == 'daily':
            params = {
                'n_neighbors': 15,  # Arttırıldı
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 30
            }
        else:  # hourly
            params = {
                'n_neighbors': 25,  # Arttırıldı
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 30
            }
        
        # Model oluştur ve eğit
        print(f"\n{symbol} ({interval}) modeli eğitiliyor...")
        model = KNeighborsRegressor(**params)
        model.fit(X_train_scaled, y_train)
        
        # Performans hesapla
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        # MAE ve RMSE hesapla (test verisi için)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Modeli kaydet
        model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
        joblib.dump(model, model_path)
        
        # Metrikleri kaydet
        performance = {
            'symbol': symbol,
            'interval': interval,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'features': list(features.columns),
            'scaler': scaler
        }
        
        performance_path = os.path.join(model_dir, f"{symbol}_performance.joblib")
        joblib.dump(performance, performance_path)
        
        # Sonuçları göster
        print(f"\nModel kaydedildi: {model_path}")
        print(f"Performans metrikleri kaydedildi: {performance_path}")
        print(f"Eğitim R2 skoru: {r2_train:.4f}")
        print(f"Test R2 skoru: {r2_test:.4f}")
        print(f"Test MAE: {mae_test:.4f}")
        print(f"Test RMSE: {rmse_test:.4f}")
        
        return {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'scaler': scaler
        }
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return None

def main():
    # Desteklenen kriptolar
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK',
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    
    # Zaman aralıkları
    intervals = ['hourly', 'daily', 'monthly']
    
    # Her zaman aralığı için modelleri eğit
    for interval in intervals:
        print(f"\n{interval.upper()} modelleri eğitiliyor...")
        for symbol in symbols:
            train_model(symbol, interval)

if __name__ == "__main__":
    main()
