import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class DTRModel:
    def __init__(self, interval):
        # Model parametrelerinin belirlenmesi
        if interval.lower() == 'monthly':
            params = {
                'max_depth': 4,
                'min_samples_split': 4,
                'min_samples_leaf': 3
            }
        elif interval.lower() == 'hourly':
            params = {
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            }
        else:  
            params = {
                'max_depth': 5,
                'min_samples_split': 8,
                'min_samples_leaf': 4
            }
        
        self.model = DecisionTreeRegressor(
            **params,
            random_state=42
        )
        self.interval = interval #Zaman aralığını saklar.
        self.is_trained = False #Modelin eğitilmediğini belirler.

    def prepare_data(self, symbol):
        
        # Verilerin hazırlanması
        
        # Veri dosyasının yolu
        data_path = f"d:/Crypto_Analysis_random_state/data/processed/{self.interval.lower()}_processed/{symbol}.xlsx"
        
        # Verilerin okunması
        df = pd.read_excel(data_path)
        
        # Bağımlı ve bağımsız değişkenlerin belirlenmesi
        y = df['close']
        X = df.drop(['timestamp', 'close'], axis=1)
        
        # Verilerin eğitim ve test kümelerine bölünmesi
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, X.columns

    def train(self, symbol):
        
        # Modelin eğitilmesi
    
        print(f"\n{symbol} ({self.interval}) modeli eğitiliyor...")
        
        # Verilerin hazırlanması
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(symbol)
        
        # Modelin eğitilmesi
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Tahminlerin yapılması
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Performans ölçümlerinin hesaplanması
        mae_test = mean_absolute_error(y_test, test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
        
        performance = {
            'symbol': symbol,
            'interval': self.interval,
            'r2_train': r2_score(y_train, train_pred),
            'r2_test': r2_score(y_test, test_pred),
            'mse_train': mean_squared_error(y_train, train_pred),
            'mse_test': mean_squared_error(y_test, test_pred),
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'features': list(feature_names),
            'feature_importance': self._get_feature_importance(feature_names)
        }
        
        # Modelin ve performans ölçümlerinin kaydedilmesi
        self._save_results(symbol, performance)
        
        # Performans ölçümlerinin gösterilmesi
        self._print_results(performance)
        
        return performance

    def _get_feature_importance(self, feature_names):
        
        # Özellik önem derecelerinin hesaplanması
        # Karar ağacı modelleri, bazı özelliklere daha fazla önem vererek tahmin yapar.
        # Bu fonksiyon, hangi özelliklerin (feature) model için en önemli olduğunu belirler.
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.to_dict('records')

    def _save_results(self, symbol, performance):
        
        # Modelin ve performans ölçümlerinin kaydedilmesi
        
        # Model klasörünün oluşturulması
        model_dir = f"d:/Crypto_Analysis_random_state/models/decision_tree/{self.interval.lower()}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Modelin kaydedilmesi
        model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
        joblib.dump(self.model, model_path)
        
        # Performans ölçümlerinin kaydedilmesi
        performance_path = os.path.join(model_dir, f"{symbol}_performance.joblib")
        joblib.dump(performance, performance_path)

    def _print_results(self, performance):
        
        # Performans ölçümlerinin gösterilmesi
        
        print("\n" + "="*50)
        print(f"{performance['symbol']} ({performance['interval']}) modeli sonuçları")
        print("="*50)
        print(f"▪ R² (eğitim): {performance['r2_train']:.4f}")
        print(f"▪ R² (test): {performance['r2_test']:.4f}")
        print(f"▪ MSE (eğitim): {performance['mse_train']:.4f}")
        print(f"▪ MSE (test): {performance['mse_test']:.4f}")
        print(f"▪ MAE (test): {performance['mae_test']:.4f}")
        print(f"▪ RMSE (test): {performance['rmse_test']:.4f}")
        print("\n▪ Önemli 5 özellik:")
        importance_df = pd.DataFrame(performance['feature_importance'])
        print(importance_df.head().to_string())
        print("="*50 + "\n")

def train_all_models():
    
    # Tüm kripto paralar için DTR modellerinin eğitilmesi
    
    # Desteklenen kripto para listesi
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK',
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
               
    # Desteklenen zaman aralıkları
    intervals = ['Daily', 'Monthly', 'Hourly']
    
    # Her zaman aralığı için modellerin eğitilmesi
    for interval in intervals:
        print(f"\n{interval} DTR modelleri eğitiliyor...")
        model = DTRModel(interval)
        for symbol in symbols:
            try:
                model.train(symbol)
            except Exception as e:
                print(f"Hata: {symbol} ({interval}) eğitilirken bir hata oluştu:")
                print(str(e))

if __name__ == "__main__":
    train_all_models()
