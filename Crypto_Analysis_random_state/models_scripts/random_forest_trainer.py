import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Kripto para fiyat tahmini için Random Forest Regressor modellerini eğiten modül.
def train_model(symbol, interval):
    try:
        
        base_dir = "d:/Crypto_Analysis_random_state"
        data_dir = os.path.join(base_dir, "data", "processed", f"{interval}_processed")
        model_dir = os.path.join(base_dir, "models", "random_forest", interval)
        
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
        
        # Veri ölçekleme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model parametreleri
        if interval.lower() == 'monthly':
            params = {
                'n_estimators': 200,           # Ağaç sayısı
                'max_depth': 3,                # Derinlik
                'min_samples_split': 3,        # Bölme için örnek
                'min_samples_leaf': 2,         # Yaprak için örnek
                'max_features': 0.8,           # Özellik oranı
                'bootstrap': True,             # Bootstrap Sampling :Her ağaç için veri setinden rastgele örnekler alınır
                'oob_score': True,             # OOB (Out-of-Bag) :erileriyle test edilerek performans ölçülebilir Avantajı: Ekstra bir test seti kullanmadan genel başarıyı tahmin etmeye yardımcı olur.
                'warm_start': True             # Modeli Yeniden Eğitmeye İzin Verir
            }
        elif interval.lower() == 'hourly':
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 0.8,
                'bootstrap': True,
                'oob_score': True
            }
        else:  # daily
            params = {
                'n_estimators': 400,
                'max_depth': 5,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'max_features': 0.75,
                'bootstrap': True,
                'oob_score': True
            }
        
        # Model oluştur
        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1  # Tüm CPU'ları kullan
        )
        
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
        
        # Özellik önemi
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
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
    
    # Zaman aralıkları
    intervals = ['Monthly', 'Daily', 'Hourly']
    
    # Modelleri eğit
    for interval in intervals:
        print(f"\n{interval} modelleri eğitiliyor...")
        print("=" * 50)
        
        for symbol in symbols:
            print(f"\n{symbol} ({interval}) işleniyor...")
            train_model(symbol, interval)
