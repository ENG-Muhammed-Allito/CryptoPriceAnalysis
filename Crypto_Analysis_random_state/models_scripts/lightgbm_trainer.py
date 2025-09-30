import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

def train_model(df, symbol, interval, model_dir):
    
    #LightGBM modeli eğitir ve performans metriklerini hesaplar.
    
    try:
        # Hedef ve özellikler
        target = df['close']
        features = df.drop(['timestamp', 'close'], axis=1, errors='ignore')
        
        # Sayısal olmayan sütunları kontrol et ve çıkar
        numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
        features = features[numeric_features]
        
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
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model parametreleri
        if interval.lower() == 'monthly':
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1  # Tüm CPU'ları kullan
            }
        elif interval.lower() == 'daily':
            params = {
                'n_estimators': 300,
                'max_depth': 7,
                'learning_rate': 0.03,
                'num_leaves': 63,
                'min_child_samples': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1  # Tüm CPU'ları kullan
            }
        else:  # hourly
            params = {
                'n_estimators': 500,
                'max_depth': 9,
                'learning_rate': 0.01,
                'num_leaves': 127,
                'min_child_samples': 20,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1  # Tüm CPU'ları kullan
            }
        
        # Model oluştur ve eğit
        print(f"\n{symbol} ({interval}) modeli eğitiliyor...")
        model = lgb.LGBMRegressor(**params)
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
        
        # Metrikleri kaydet
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
        print(f"Hata: {str(e)}")
        return None

def main():
    # Ana dizin
    base_dir = "d:/Crypto_Analysis_random_state"
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models/lightgbm")
    
    # Desteklenen kriptolar
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK',
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    
    # Zaman aralıkları
    intervals = {
        'HOURLY': ('Hourly_processed', 'hourly'),
        'DAILY': ('Daily_processed', 'daily'),
        'MONTHLY': ('Monthly_processed', 'monthly')
    }
    
    for interval_key, (data_folder, interval) in intervals.items():
        print(f"\n{interval_key} modelleri eğitiliyor...")
        
        # Dizinleri oluştur
        interval_dir = os.path.join(models_dir, interval.lower())
        os.makedirs(interval_dir, exist_ok=True)
        
        # Veri dizini
        data_folder_path = os.path.join(data_dir, data_folder)
        
        if os.path.exists(data_folder_path):
            # Her kripto para için model eğit
            for symbol in symbols:
                data_file = os.path.join(data_folder_path, f"{symbol}.xlsx")
                if os.path.exists(data_file):
                    # Veriyi oku
                    df = pd.read_excel(data_file)
                    train_model(df, symbol, interval, interval_dir)
                else:
                    print(f"{symbol} için veri bulunamadı: {data_file}")
        else:
            print(f"{data_folder_path} bulunamadı.")

if __name__ == "__main__":
    main()
