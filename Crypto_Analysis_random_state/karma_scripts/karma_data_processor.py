import pandas as pd
import numpy as np
import os
import joblib
import sys

sys.path.append('d:/Crypto_Analysis_random_state')
from normalizer import CryptoNormalizer

# Karma veri setini işleyen ve teknik göstergeleri hesaplayan ana modül
def calculate_technical_indicators(df, interval):
    # Teknik göstergeleri hesaplar ve DataFrame'e ekler
    df_copy = df.copy()
    
    # Zaman aralığına göre pencere ayarları
    if interval.lower() == 'monthly':
        ma_windows = [3, 6, 12]  # Aylık pencereler
        volatility_window = 6    
        rsi_window = 6          
        macd_windows = [3, 6, 2] 
        bb_window = 6
        roc_window = 6
        atr_window = 6
    elif interval.lower() == 'hourly':
        ma_windows = [24, 48, 168]  # Saatlik pencereler
        volatility_window = 24      
        rsi_window = 24            
        macd_windows = [12, 26, 9]  
        bb_window = 24
        roc_window = 24
        atr_window = 24
    else:  # daily
        ma_windows = [7, 14, 30]    # Günlük pencereler
        volatility_window = 14      
        rsi_window = 14            
        macd_windows = [12, 26, 9]  
        bb_window = 20
        roc_window = 14
        atr_window = 14
    
    # Sembol bazında gruplama
    grouped = df_copy.groupby('symbol')
    
    # Yeni bir DataFrame oluştur
    result_dfs = []
    
    for symbol, group in grouped:
        # Grubu timestamp'e göre sırala
        if 'timestamp' in group.columns:
            group = group.sort_values('timestamp')
        
        # Hareketli ortalamalar
        if len(group) >= ma_windows[2]:  # En az büyük pencere kadar veri noktası olmalı
            group['MA7'] = group['close'].rolling(window=ma_windows[0]).mean()
            group['MA14'] = group['close'].rolling(window=ma_windows[1]).mean()
            group['MA30'] = group['close'].rolling(window=ma_windows[2]).mean()
            
            # Volatilite hesaplama
            group['volatility'] = group['close'].rolling(window=volatility_window).std() / group['close'].rolling(window=volatility_window).mean()
            
            # RSI (Relative Strength Index)
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
            rs = gain / loss
            group['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema12 = group['close'].ewm(span=macd_windows[0], adjust=False).mean()
            ema26 = group['close'].ewm(span=macd_windows[1], adjust=False).mean()
            group['MACD'] = ema12 - ema26
            group['Signal_Line'] = group['MACD'].ewm(span=macd_windows[2], adjust=False).mean()
            
            # Bollinger Bantları
            group['BB_middle'] = group['close'].rolling(window=bb_window).mean()
            group['BB_upper'] = group['BB_middle'] + 2 * group['close'].rolling(window=bb_window).std()
            group['BB_lower'] = group['BB_middle'] - 2 * group['close'].rolling(window=bb_window).std()
            
            # Price Rate of Change (ROC)
            group['ROC'] = group['close'].pct_change(periods=roc_window) * 100
            
            # Average True Range (ATR)
            high_low = group['high'] - group['low']
            high_close = (group['high'] - group['close'].shift()).abs()
            low_close = (group['low'] - group['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            group['ATR'] = tr.rolling(window=atr_window).mean()
        
        # Eksik değerleri doldur
        group = group.ffill().bfill()
        
        # Sonuç listesine ekle
        result_dfs.append(group)
    
    # Tüm sonuçları birleştir
    df_result = pd.concat(result_dfs, ignore_index=True)
    
    return df_result

def process_karma_data(interval):
    # Karma veri setini işler ve teknik göstergeleri ekler
    try:
        print(f"\nKarma {interval} verileri işleniyor...")
        
        # Veri klasörlerini belirle
        input_file = f"d:/Crypto_Analysis_random_state/Karma_data/{interval.capitalize()}_data/all_crypto_{interval}.xlsx"
        output_dir = f"d:/Crypto_Analysis_random_state/Karma_data/{interval.capitalize()}_data/processed"
        normalizer_dir = f"d:/Crypto_Analysis_random_state/models/normalizers/karma_{interval}"
        
        # Çıktı klasörleri kontrolü
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(normalizer_dir, exist_ok=True)
        
        # Veri okuma
        print(f"{interval.capitalize()} veri seti yükleniyor...")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Dosya bulunamadı: {input_file}")
        
        df = pd.read_excel(input_file)
        print(f"Veri seti yüklendi. Boyut: {df.shape}")
        
        # Veri temizleme
        print("Veri temizleme işlemi başlatılıyor...")
        
        # Eksik değerleri kontrol et
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Eksik değerler bulundu: {missing_values[missing_values > 0]}")
            # İleri yönlü doldurma (Forward fill)
            df = df.ffill()
            # Geriye yönlü doldurma (Backward fill)
            df = df.bfill()
            print("Eksik değerler dolduruldu.")
        else:
            print("Eksik değer bulunamadı.")
        
        # Aykırı değerleri işle (IQR yöntemi)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col != 'symbol':  # symbol sütununu atla
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Aykırı değerleri sınırlara çek
                df[col] = np.where(
                    df[col] < lower_bound,
                    lower_bound,
                    np.where(
                        df[col] > upper_bound,
                        upper_bound,
                        df[col]
                    )
                )
        
        print("Aykırı değerler işlendi.")
        
        # Gösterge hesaplama
        print("Teknik göstergeler ekleniyor...")
        df = calculate_technical_indicators(df, interval)
        print(f"Teknik göstergeler eklendi. Yeni boyut: {df.shape}")
        
        # Eksik veri temizleme
        df = df.dropna().reset_index(drop=True)
        
        # Saatlik veriler için veri boyutunu azalt
        if interval.lower() == 'hourly' and len(df) > 20000:
            df = df.sample(n=20000, random_state=42)
            print(f"Veri boyutu azaltıldı: {len(df)} satır")
        
        # Verileri karıştır (shuffle) - Rastgele sıralama
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Veriler rastgele karıştırıldı.")
        
        # Normalizasyon
        print("Veriler normalize ediliyor...")
        
        # CryptoNormalizer ile normalizasyon (Min-Max Scaling)
        normalizer = CryptoNormalizer()
        
        # 'symbol' ve 'timestamp' sütunlarını ayır
        non_numeric_cols = ['symbol']
        if 'timestamp' in df.columns:
            non_numeric_cols.append('timestamp')
        
        # Sayısal sütunları seç
        numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
        
        # Veriyi normalize et
        df_normalized = normalizer.fit_transform(df, columns=numeric_cols)
        
        # Normalizer'ı kaydet
        normalizer_path = os.path.join(normalizer_dir, f"karma_{interval}_normalizer.joblib")
        joblib.dump(normalizer, normalizer_path)
        print(f"Normalizer kaydedildi: {normalizer_path}")
        
        # İşlenmiş veriyi kaydet
        processed_data_path = os.path.join(output_dir, f"all_processed_{interval}.xlsx")
        df_normalized.to_excel(processed_data_path, index=False)
        print(f"İşlenmiş veri kaydedildi: {processed_data_path}")
        
        print(f"Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
        print(f"Veri setindeki kripto para sayısı: {df['symbol'].nunique()}")
        
        print("Veriler başarıyla kaydedildi.")
        return True
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

def process_all_intervals():
    # Tüm zaman aralıkları için veri işleme sürecini çalıştırır
    print("\nKarma veri işleme süreci başlatılıyor...")
    
    intervals = ['daily', 'hourly', 'monthly']
    results = {}
    
    for interval in intervals:
        results[interval] = process_karma_data(interval)
    
    print("\nİşlem sonuçları:")
    for interval, success in results.items():
        print(f"- {interval.capitalize()}: {'Başarılı' if success else 'Başarısız'}")
    
    print("Karma veri işleme süreci tamamlandı!")

if __name__ == "__main__":
    process_all_intervals()
