import pandas as pd
import numpy as np
from normalizer import CryptoNormalizer
import os

# Kripto para verilerini işleyen ve teknik göstergeleri hesaplayan ana modül.
def calculate_technical_indicators(df, interval):
    # Teknik göstergeleri hesaplar ve DataFrame'e ekler.
    df_copy = df.copy()
    
    # Zaman aralığına göre pencere ayarları
    if interval.lower() == 'monthly':
        ma_windows = [3, 6, 12]  # Aylık pencereler
        volatility_window = 6    
        rsi_window = 6          
        macd_windows = [3, 6, 2] 
        bb_window = 6           
    elif interval.lower() == 'hourly':
        ma_windows = [24, 48, 168]  # Saatlik pencereler
        volatility_window = 24      
        rsi_window = 24            
        macd_windows = [12, 26, 9]  
        bb_window = 24             
    else:  # daily
        ma_windows = [7, 14, 30]    # Günlük pencereler
        volatility_window = 14      
        rsi_window = 14            
        macd_windows = [12, 26, 9]  
        bb_window = 20             
    
    #Feature Engineering
    #Amaç: Ham verilerden faydalı göstergeler oluşturmak.
    
    # (MA) Hareketli ortalamalar hesaplama
    #Amaç: Fiyat trendlerini belirlemek.
    df_copy['MA7'] = df_copy['close'].rolling(window=ma_windows[0]).mean()
    df_copy['MA14'] = df_copy['close'].rolling(window=ma_windows[1]).mean()
    df_copy['MA30'] = df_copy['close'].rolling(window=ma_windows[2]).mean()
    
    # Volatilite hesaplama (Kapanış fiyatlarının standart sapması alınarak volatilite ölçülür.)
    #Amaç: Fiyat oynaklığını ölçmek.
    df_copy['volatility'] = df_copy['close'].rolling(window=volatility_window).std()
    
    # RSI (Göreceli Güç Endeksi) Hesaplama
    #Amaç: Fiyatın aşırı alım/aşırı satım bölgesinde olup olmadığını belirlemek.
    delta = df_copy['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD ve Sinyal Çizgisi Hesaplama
    #Amaç: Trend değişimlerini tespit etmek.
    exp1 = df_copy['close'].ewm(span=macd_windows[0], adjust=False).mean()
    exp2 = df_copy['close'].ewm(span=macd_windows[1], adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=macd_windows[2], adjust=False).mean()
    
    # Bollinger bantları hesaplama
    #Amaç:Fiyatın aşırı değerlenip değerlenmediğini ölçmek.
    df_copy['BB_middle'] = df_copy['close'].rolling(window=bb_window).mean()
    df_copy['BB_upper'] = df_copy['BB_middle'] + 2 * df_copy['close'].rolling(window=bb_window).std()
    df_copy['BB_lower'] = df_copy['BB_middle'] - 2 * df_copy['close'].rolling(window=bb_window).std()
    
    # Zaman özellikleri eklenir
    #Amaç: Fiyatın zaman içindeki trendini tespit etmek.
    
    return df_copy

def process_crypto_data(symbol, interval):
    # Kripto para verilerini işler ve teknik göstergeleri ekler.
    #Amaç: Veri ön işleme ve teknik göstergeleri hesaplama.
    try:
        print(f"\n{symbol} ({interval}) işleniyor...")
        
        # Veri klasörlerini belirle
        input_dir = f"d:/Crypto_Analysis_random_state/data/{interval}_data"
        output_dir = f"d:/Crypto_Analysis_random_state/data/processed/{interval}_processed"
        
        # Çıktı klasörü kontrolü
        os.makedirs(output_dir, exist_ok=True)
        
        # Veri okuma
        input_path = os.path.join(input_dir, f"{symbol}.xlsx")
        df = pd.read_excel(input_path)
        
        # Gösterge hesaplama
        #Amaç: Veri ön işleme ve teknik göstergeleri hesaplama.
        df = calculate_technical_indicators(df, interval)
        
        # Eksik veri temizleme
        df = df.dropna().reset_index(drop=True)
        
        # Normalizasyon
        #Amaç: Veri ölçeklendirme.
        normalizer = CryptoNormalizer()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_normalized = normalizer.fit_transform(df, columns=numeric_columns)
        
        # Model kaydetme
        #Amaç: Veri normalizasyon modelini kaydetme.
        normalizer.save(f"{symbol}_{interval}")
        
        # Sonuç kaydetme
        output_path = os.path.join(output_dir, f"{symbol}.xlsx")
        df_normalized.to_excel(output_path, index=False)
        
        print(f"Veriler işlendi ve kaydedildi: {output_path}")
        print(f"Satır sayısı: {len(df_normalized)}")
        
    except Exception as e:
        print(f"Hata: {symbol} ({interval}) işlenirken bir hata oluştu:")
        print(str(e))

def process_all_cryptos():
    # Tüm kripto paraları işler.
    #Amaç: Tüm kripto paralarını işlemek.

    # Kripto listesi
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK', 
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    
    # Zaman aralıkları
    intervals = ['daily', 'monthly', 'hourly']
    
    # İşleme başla
    for interval in intervals:
        print(f"\n{interval} verileri işleniyor...")
        for symbol in symbols:
            process_crypto_data(symbol, interval)

if __name__ == "__main__":
    process_all_cryptos()
