import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Kripto para fiyat tahmini için Linear Regression modellerini eğiten modül.
def train_model(symbol, interval):
    try:
        base_dir = "d:/Crypto_Analysis_random_state"
        data_dir = os.path.join(base_dir, "data", "processed", f"{interval}_processed")
        model_dir = os.path.join(base_dir, "models", "ridge_regression", interval)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Veriyi yükle
        data_file = os.path.join(data_dir, f"{symbol}.xlsx")
        if not os.path.exists(data_file):
            print(f"Veri dosyası bulunamadı: {data_file}")
            return None
        
        df = pd.read_excel(data_file)
        
        # Hedef ve özellikler
        target = df['close']
        features = df.drop(['timestamp', 'close'], axis=1)
        
        # Veri bölme
        if interval.lower() == 'monthly':
            # Aylık: %15 test
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.15, random_state=42
            )
        else:
            # Diğer: %20 test
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
        
        # Model oluştur
        if interval.lower() == 'monthly':
            # Aylık veriler için parametreler
            model = Ridge(
                alpha=10.0,          # Daha güçlü regularization
                fit_intercept=True,
                max_iter=2000,
                tol=0.0001,         # Daha hassas yakınsama
                random_state=42,
                solver='sag'         # Stochastic Average Gradient - büyük veri setleri için daha iyi
            )
        elif interval.lower() == 'hourly':
            # Saatlik veriler için çok daha güçlü regularization
            model = Ridge(
                alpha=500.0,         # Çok daha güçlü regularization
                fit_intercept=True,
                max_iter=5000,       # Daha fazla iterasyon
                tol=0.00001,         # Daha hassas yakınsama
                random_state=42,
                solver='saga'        # En hızlı solver
            )
        else:  # daily
            # Günlük veriler için parametreler
            model = Ridge(
                alpha=100.0,         # Daha güçlü regularization
                fit_intercept=True,
                max_iter=3000,
                tol=0.00001,
                random_state=42,
                solver='sag'
            )
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Eğitim
        print(f"\n{symbol} ({interval}) modeli eğitiliyor...")
        model.fit(X_train_scaled, y_train)
        
        # Performans hesapla
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        # MAE ve RMSE hesapla (test verisi için)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Özellik önemi (katsayıların mutlak değerleri)
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': np.abs(model.coef_)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Modeli kaydet
        model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
        joblib.dump(model, model_path)
        
        # Performans bilgilerini kaydet
        performance = {
            'symbol': symbol,
            'interval': interval,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'features': list(features.columns),
            'feature_importance': feature_importance.to_dict('records'),
            'scaler': scaler  # Ölçeklendirme parametrelerini de kaydediyoruz
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
        print("\nÖzellik önem sıralaması:")
        print(feature_importance.head())
        
        return {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'feature_importance': feature_importance,
            'scaler': scaler
        }
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return None

if __name__ == "__main__":
    # Desteklenen kriptolar
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK',
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    
    # Her sembol ve zaman aralığı için modeli eğit
    intervals = ['hourly', 'daily', 'monthly']
    
    for interval in intervals:
        print(f"\n{interval.upper()} modelleri eğitiliyor...")
        for symbol in symbols:
            train_model(symbol, interval)
