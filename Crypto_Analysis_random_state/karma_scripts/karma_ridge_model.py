import os
import joblib
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Normalizer sınıfına erişim için sistem yolunu ekle
sys.path.append('d:/Crypto_Analysis_random_state')
from normalizer import CryptoNormalizer

class KarmaRidgeModel:
    
    def __init__(self, interval):
        
        self.interval = interval.lower()
        
        # Model parametrelerinin belirlenmesi
        if self.interval == 'monthly':
            params = {
                'alpha': 500.0,
                'fit_intercept': True,
                'copy_X': True,
                'max_iter': None,
                'tol': 0.001,
                'solver': 'auto',
                'random_state': 42
            }
        elif self.interval == 'hourly':
            params = {
                'alpha': 10000.0,
                'fit_intercept': True,
                'copy_X': True,
                'max_iter': None,
                'tol': 0.001,
                'solver': 'auto',
                'random_state': 42
            }
        else:  # daily
            params = {
                'alpha': 10000.0,
                'fit_intercept': True,
                'copy_X': True,
                'max_iter': None,
                'tol': 0.001,
                'solver': 'auto',
                'random_state': 42
            }
        
        self.model = Ridge(**params)
        
        # Veri ve model yolları - İşlenmiş veri yolunu kullan
        self.data_path = f"d:/Crypto_Analysis_random_state/Karma_data/{self.interval.capitalize()}_data/processed/all_processed_{self.interval}.xlsx"
        self.model_dir = f"d:/Crypto_Analysis_random_state/karma_models/ridge/{self.interval.lower()}"
        self.normalizer_path = f"d:/Crypto_Analysis_random_state/models/normalizers/karma_{self.interval}/karma_{self.interval}_normalizer.joblib"
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.is_trained = False
        self.normalizer = None
    
    def prepare_data(self):
        # Karma veri setini hazırlar
        print(f"\n{self.interval.capitalize()} Ridge modeli için veriler hazırlanıyor...")
        
        # Veriyi yükle
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.data_path}")
            
        df = pd.read_excel(self.data_path)
        
        # Normalizer'ı yükle
        if os.path.exists(self.normalizer_path):
            self.normalizer = joblib.load(self.normalizer_path)
            print(f"Normalizer yüklendi: {self.normalizer_path}")
        else:
            print(f"Uyarı: Normalizer dosyası bulunamadı: {self.normalizer_path}")
            print("Veri zaten normalize edilmiş olarak kabul ediliyor.")
        
        # Güvenlik kontrolü: Eksik değer kontrolü
        if df.isnull().sum().sum() > 0:
            print("Uyarı: İşlenmiş veride eksik değerler bulundu. Otomatik olarak doldurulacak.")
            # İleri yönlü doldurma
            df = df.ffill()
            # Geriye yönlü doldurma
            df = df.bfill()
            # Kalan eksik değerleri 0 ile doldurma
            df = df.fillna(0)
        
        # Hedef ve özellikler
        y = df['close']
        X = df.drop(['timestamp', 'close', 'symbol'], axis=1, errors='ignore')
        
        # Verilerin eğitim ve test kümelerine bölünmesi
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, X.columns
    
    def train(self):
        # Modeli eğitir
        print(f"\n{self.interval.capitalize()} Ridge modeli eğitiliyor...")
        
        # Verilerin hazırlanması
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data()
        
        # Modelin eğitilmesi
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Tahminlerin yapılması
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Performans ölçümlerinin hesaplanması
        performance = {
            'interval': self.interval,
            'r2_train': r2_score(y_train, train_pred),
            'r2_test': r2_score(y_test, test_pred),
            'mse_train': mean_squared_error(y_train, train_pred),
            'mse_test': mean_squared_error(y_test, test_pred),
            'mae_test': mean_absolute_error(y_test, test_pred),
            'rmse_test': np.sqrt(mean_squared_error(y_test, test_pred)),
            'features': list(feature_names)
        }
        
        # Sonuçların kaydedilmesi ve gösterilmesi
        self._save_results(performance)
        self._print_results(performance)
        
        return performance
    
    def _save_results(self, performance):
        # Modelin ve performans ölçümlerinin kaydedilmesi
        # Model dosyasını kaydet
        model_file = os.path.join(self.model_dir, "karma_model.joblib")
        joblib.dump(self.model, model_file)
        print(f"Model kaydedildi: {model_file}")
        
        # Performans metriklerini kaydet
        performance_file = os.path.join(self.model_dir, "karma_performance.joblib")
        joblib.dump(performance, performance_file)
        print(f"Performans metrikleri kaydedildi: {performance_file}")
    
    def _print_results(self, performance):
        # Performans ölçümlerinin gösterilmesi
        print("\n" + "=" * 50)
        print(f"Karma {self.interval.capitalize()} Ridge modeli sonuçları")
        print("=" * 50)
        print(f"▪ R² (eğitim): {performance['r2_train']:.6f}")
        print(f"▪ R² (test): {performance['r2_test']:.6f}")
        print(f"▪ MSE (eğitim): {performance['mse_train']:.6f}")
        print(f"▪ MSE (test): {performance['mse_test']:.6f}")
        print(f"▪ MAE (test): {performance['mae_test']:.6f}")
        print(f"▪ RMSE (test): {performance['rmse_test']:.6f}")
        print("=" * 50)
        
        # Özellik katsayılarını göster
        coefficients = sorted(zip(performance['features'], self.model.coef_), 
                             key=lambda x: abs(x[1]), reverse=True)
        for feature, coef in coefficients[:5]:  # En önemli 5 özelliği göster
            print(f"  - {feature}: {coef:.4f}")
        
        print()

def train_all_intervals():
    # Tüm zaman aralıkları için Ridge modellerinin eğitilmesi
    intervals = ['daily', 'hourly', 'monthly']
    results = {}
    
    for interval in intervals:
        try:
            model = KarmaRidgeModel(interval)
            performance = model.train()
            results[interval] = performance['r2_test']
        except Exception as e:
            print(f"Hata ({interval}): {str(e)}")
            results[interval] = None
    
    print("\nEğitim sonuçları:")
    for interval, r2 in results.items():
        if r2 is not None:
            print(f"▪ {interval.capitalize()}: R² = {r2:.6f}")
        else:
            print(f"▪ {interval.capitalize()}: Başarısız")
    
    print("\nKarma Ridge modelleri eğitimi tamamlandı!")

if __name__ == "__main__":
    train_all_intervals()
