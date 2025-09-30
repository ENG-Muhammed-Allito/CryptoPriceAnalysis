import os
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# SVR modeli sınıfı.
class SVRModel:
    def __init__(self, interval):
        # SVR model parametreleri.
        #kernel: Çekirdeğin türü (rbf).veriyi daha yüksek boyutlu bir uzaya dönüştürerek daha karmaşık ilişkileri modellememize yardımcı olur.
        #C: Modelin ceza parametresidir.
        #epsilon: Modelin hata toleransıdır.
        #gamma:  Kernel fonksiyonunun etki alanını belirler
        #tol: Optimizasyon sırasında kullanılan tolerans değeridir
        #cache_size: Modelin bellekte tutabileceği maksimum bellek boyutudur (MB cinsinden).
        if interval.lower() == 'monthly':
            self.params = {'kernel': 'rbf', 'C': 100, 'epsilon': 0.1, 'gamma': 'scale', 'tol': 1e-3, 'cache_size': 1000}
        elif interval.lower() == 'hourly':
            self.params = {'kernel': 'rbf', 'C': 1000, 'epsilon': 0.01, 'gamma': 'scale', 'tol': 1e-4, 'cache_size': 1000}
        else:  
            self.params = {'kernel': 'rbf', 'C': 500, 'epsilon': 0.05, 'gamma': 'scale', 'tol': 1e-3, 'cache_size': 1000}
        
        self.model = SVR(**self.params)
        self.interval = interval
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_data(self, symbol):
        # Veri hazırlama ve ölçeklendirme.
        try:
            data_path = f"d:/Crypto_Analysis_random_state/data/processed/{self.interval}_processed/{symbol}.xlsx"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")
            df = pd.read_excel(data_path)
            features = df.drop(['timestamp', 'close'], axis=1)
            target = df['close']
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"Veri hazırlama hatası ({symbol}): {str(e)}")
            raise

    def train(self, symbol):
        # Model eğitimi ve performans hesaplama.
        try:
            print(f"\n{symbol} için {self.interval} SVR modeli eğitiliyor...")
            X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(symbol)
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Performans hesapla
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # MAE ve RMSE hesapla (test verisi için)
            mae_test = mean_absolute_error(y_test, test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Performans bilgilerini kaydet
            performance = {'symbol': symbol, 
                          'interval': self.interval, 
                          'r2_train': r2_score(y_train, train_pred), 
                          'r2_test': r2_score(y_test, test_pred), 
                          'mse_test': mean_squared_error(y_test, test_pred),
                          'mae_test': mae_test,
                          'rmse_test': rmse_test}
            
            self._print_results(performance)
            self._save_model(symbol, performance)
            return performance
            
        except Exception as e:
            print(f"Model eğitimi hatası ({symbol}): {str(e)}")
            raise

    def _save_model(self, symbol, performance):
        # Model ve performans sonuçlarını kaydetme.
        try:
            model_dir = f"d:/Crypto_Analysis_random_state/models/svr/{self.interval}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
            joblib.dump((self.model, self.scaler), model_path)
            performance_path = os.path.join(model_dir, f"{symbol}_performance.joblib")
            joblib.dump(performance, performance_path)
            print(f"Model kaydedildi: {model_path}")
            print(f"Performans metrikleri kaydedildi: {performance_path}")
            
        except Exception as e:
            print(f"Model kaydetme hatası ({symbol}): {str(e)}")
            raise

    def _print_results(self, performance):
        # Model sonuçlarını ekrana yazdırma.
        print(f"\n{performance['symbol']} için {performance['interval']} SVR Modeli Sonuçları:")
        print("-" * 50)
        print(f"Eğitim R² Skoru: {performance['r2_train']:.4f}")
        print(f"Test R² Skoru: {performance['r2_test']:.4f}")
        print(f"Test MSE: {performance['mse_test']:.4f}")
        print(f"Test MAE: {performance['mae_test']:.4f}")
        print(f"Test RMSE: {performance['rmse_test']:.4f}")
        print("-" * 50)

def train_all_models():
    # Tüm kripto paralar için SVR modellerini eğitme.
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK', 'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    intervals = ['Daily', 'Monthly', 'Hourly']
    for interval in intervals:
        print(f"\n{interval} SVR modelleri eğitiliyor...")
        print("=" * 50)
        model = SVRModel(interval)
        for symbol in symbols:
            try:
                model.train(symbol)
            except Exception as e:
                print(f"{symbol} için eğitim başarısız: {str(e)}")
                continue

if __name__ == "__main__":
    train_all_models()
