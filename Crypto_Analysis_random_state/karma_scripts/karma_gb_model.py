import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class KarmaGBModel:
    # Karma veri seti için Gradient Boosting Regresyon modeli
    def __init__(self, interval):
        # Karma veri seti için Gradient Boosting Regresyon modeli
        self.interval = interval.lower()
        
        # Model parametrelerinin belirlenmesi
        if self.interval == 'monthly':
            params = {
                'n_estimators': 80,
                'learning_rate': 0.015,
                'max_depth': 2,
                'min_samples_split': 25,
                'min_samples_leaf': 15,
                'subsample': 0.8,
                'max_features': 0.7,
                'random_state': 42
            }
        elif self.interval == 'hourly':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.02,
                'max_depth': 3,
                'min_samples_split': 30,
                'min_samples_leaf': 20,
                'subsample': 0.8,
                'max_features': 0.8,
                'random_state': 42
            }
        else:  # daily
            params = {
                'n_estimators': 60,
                'learning_rate': 0.015,
                'max_depth': 2,
                'min_samples_split': 35,
                'min_samples_leaf': 25,
                'subsample': 0.8,
                'max_features': 0.7,
                'random_state': 42
            }
        
        self.model = GradientBoostingRegressor(**params)
        
        # Veri ve model yolları
        self.data_path = f"d:/Crypto_Analysis_random_state/Karma_data/{self.interval.capitalize()}_data/processed/all_processed_{self.interval}.xlsx"
        self.model_dir = f"d:/Crypto_Analysis_random_state/karma_models/gradient_boosting/{self.interval.lower()}"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.is_trained = False
        self.scaler = StandardScaler()
    
    def prepare_data(self):
        # Veri hazırlama
        print(f"\n{self.interval.capitalize()} Gradient Boosting modeli için veriler hazırlanıyor...")
        
        # Veriyi yükle
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.data_path}")
            
        df = pd.read_excel(self.data_path)
        
        # Güvenlik kontrolü: Eksik değer kontrolü
        if df.isnull().sum().sum() > 0:
            print("Uyarı: İşlenmiş veride eksik değerler bulundu. Otomatik olarak doldurulacak.")
            # İleri yönlü doldurma
            df = df.ffill()
            # Geriye yönlü doldurma
            df = df.bfill()
            # Kalan eksik değerleri 0 ile doldurma
            df = df.fillna(0)
        
        # Özellik seçimi - sadece saatlik veriler için
        if self.interval == 'hourly':
            # Önemli özellikleri seç, daha az özellik kullanarak aşırı öğrenmeyi azalt
            selected_features = ['open', 'high', 'low', 'MA7', 'RSI']
            
            # Veri boyutunu azaltmak için rastgele örnekleme
            if len(df) > 10000:
                df = df.sample(n=6000, random_state=42)
                print(f"Veri boyutu azaltıldı: {len(df)} satır")
            
            # Hedef ve seçilmiş özellikler
            y = df['close']
            X = df[selected_features]
        elif self.interval == 'daily':
            # Önemli özellikleri seç, daha az özellik kullanarak aşırı öğrenmeyi azalt
            selected_features = ['open', 'high', 'low', 'MA7', 'MA14', 'RSI', 'MACD']
            
            # Hedef ve seçilmiş özellikler
            y = df['close']
            X = df[selected_features]
        else:
            # Hedef ve tüm özellikler
            y = df['close']
            X = df.drop(['timestamp', 'close', 'symbol'], axis=1, errors='ignore')
        
        # Verilerin eğitim ve test kümelerine bölünmesi
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Verileri ölçeklendir
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train(self):
        # Modeli eğit
        print(f"\n{self.interval.capitalize()} Gradient Boosting modeli eğitiliyor...")
        
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
            'features': list(feature_names),
            'scaler': self.scaler,
            'feature_importance': self.model.feature_importances_
        }
        
        # Sonuçların kaydedilmesi ve gösterilmesi
        self._save_results(performance)
        self._print_results(performance)
        
        return performance
    
    def _save_results(self, performance):
        # Modeli ve performans ölçümlerini kaydet
        model_file = os.path.join(self.model_dir, "karma_model.joblib")
        joblib.dump(self.model, model_file)
        print(f"Model kaydedildi: {model_file}")
        
        performance_file = os.path.join(self.model_dir, "karma_performance.joblib")
        joblib.dump(performance, performance_file)
        print(f"Performans metrikleri kaydedildi: {performance_file}")
    
    def _print_results(self, performance):
        # Performans ölçümlerinin gösterilmesi
        print("\n" + "=" * 50)
        print(f"Karma {self.interval.capitalize()} Gradient Boosting modeli sonuçları")
        print("=" * 50)
        print(f"▪ R² (eğitim): {performance['r2_train']:.6f}")
        print(f"▪ R² (test): {performance['r2_test']:.6f}")
        print(f"▪ MSE (eğitim): {performance['mse_train']:.6f}")
        print(f"▪ MSE (test): {performance['mse_test']:.6f}")
        print(f"▪ MAE (test): {performance['mae_test']:.6f}")
        print(f"▪ RMSE (test): {performance['rmse_test']:.6f}")
        print()
        
        # Özellik önemlerini göster
        print("▪ Önemli 10 özellik:")
        importance_df = pd.DataFrame({
            'feature': performance['features'],
            'importance': performance['feature_importance']
        })
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))
        print("=" * 50)

def train_all_intervals():
    # Tüm zaman aralıkları için Gradient Boosting modellerini eğit
    print("\nKarma Gradient Boosting modelleri eğitiliyor...")
    intervals = ['daily', 'hourly', 'monthly']
    results = {}
    
    for interval in intervals:
        try:
            model = KarmaGBModel(interval)
            performance = model.train()
            results[interval] = performance['r2_test']
        except Exception as e:
            print(f"Hata ({interval}): {str(e)}")
            results[interval] = None
    
    print("\nEğitim sonuçları:")
    for interval, r2 in results.items():
        if r2 is not None:
            print(f"- {interval.capitalize()}: R² (test) = {r2:.4f}")
        else:
            print(f"- {interval.capitalize()}: Başarısız")
    
    print("\nKarma Gradient Boosting modelleri eğitimi tamamlandı!")

if __name__ == "__main__":
    train_all_intervals()
