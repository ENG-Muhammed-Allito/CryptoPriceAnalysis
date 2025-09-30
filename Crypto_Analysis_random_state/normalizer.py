import pandas as pd
import numpy as np
import os
import joblib

# Kripto para verilerini 0-1 aralığında normalize eden sınıf.
class CryptoNormalizer:
    # Min-Max normalizasyonu kullanan ana sınıf.
    def __init__(self):
        # Özellik aralıklarını tutan sözlük yapısı.
        self.feature_ranges = {}
        
    def fit(self, df, columns=None):
        # Veri özelliklerinin min ve max değerlerini hesaplar.
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        for column in columns:
            self.feature_ranges[column] = {
                'min': df[column].min(),
                'max': df[column].max()
            }
            
        return self
        
    def transform(self, df):
        # Veriyi 0-1 aralığına normalize eder.
        df_normalized = df.copy()
        
        for column, ranges in self.feature_ranges.items():
            if column in df.columns:
                min_val = ranges['min']
                max_val = ranges['max']
                
                # Min-max hesaplama
                if max_val > min_val:
                    df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
                else:
                    df_normalized[column] = 0
                    
        return df_normalized
        
    def inverse_transform(self, df):
        # Normalize edilmiş veriyi orijinal haline döndürür.
        df_original = df.copy()
        
        for column, ranges in self.feature_ranges.items():
            if column in df.columns:
                min_val = ranges['min']
                max_val = ranges['max']
                
                # Ters çevirme
                if max_val > min_val:
                    df_original[column] = df[column] * (max_val - min_val) + min_val
                    
        return df_original
        
    def fit_transform(self, df, columns=None):
        # Fit ve transform işlemlerini birlikte yapar.
        return self.fit(df, columns).transform(df)
        
    def save(self, symbol_with_interval):
        # Normalizer modelini kaydeder.
        interval = symbol_with_interval.split('_')[1].lower()
        normalizer_dir = f"d:/Crypto_Analysis_random_state/models/normalizers/{interval}"
        os.makedirs(normalizer_dir, exist_ok=True)
        
        normalizer_path = f"d:/Crypto_Analysis_random_state/models/normalizers/{interval}/{symbol_with_interval}_normalizer.joblib"
        
        joblib.dump(self, normalizer_path)
        print(f"Normalizer modeli kaydedildi: {normalizer_path}")
        
    @classmethod
    def load(cls, symbol_with_interval):
        # Kaydedilmiş normalizer modelini yükler.
        interval = symbol_with_interval.split('_')[1].lower()
        normalizer_path = f"d:/Crypto_Analysis_random_state/models/normalizers/{interval}/{symbol_with_interval}_normalizer.joblib"
        
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {normalizer_path}")
            
        return joblib.load(normalizer_path)
