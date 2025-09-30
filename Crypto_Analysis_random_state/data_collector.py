import pandas as pd
from binance.client import Client
import os
from datetime import datetime
import time

def get_historical_data(symbol, interval, start_date="2017-01-01"):
    #Binance'den kripto para verilerini belirtilen tarihten itibaren çeker ve Excel'e kaydeder.
    try:
        # API bağlantısı kur
        client = Client(None, None)
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        
        # Zaman aralığını ayarla
        interval_mapping = {
            'daily': Client.KLINE_INTERVAL_1DAY,
            'monthly': Client.KLINE_INTERVAL_1MONTH,
            'hourly': Client.KLINE_INTERVAL_1HOUR
        }
        
        # Verileri çek
        klines = client.get_historical_klines(
            symbol=f"{symbol}USDT",
            interval=interval_mapping[interval],
            start_str=start_ts
        )
        
        # DataFrame oluştur
        df = pd.DataFrame([
            [
                int(x[0]),    # zaman
                float(x[1]),  # açılış
                float(x[2]),  # yüksek
                float(x[3]),  # düşük
                float(x[4]),  # kapanış
                float(x[5])   # hacim
            ] for x in klines
        ], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Zamanı düzelt
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Klasör yolunu belirle
        folder_mapping = {
            'daily': 'Daily_data',
            'monthly': 'Monthly_data',
            'hourly': 'Hourly_data'
        }
        
        # Dosya yolunu oluştur
        base_dir = "d:/Crypto_Analysis_random_state/data"
        data_dir = os.path.join(base_dir, folder_mapping[interval])
        
        # Klasörü kontrol et
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Excel'e kaydet
        output_path = os.path.join(data_dir, f"{symbol}.xlsx")
        df.to_excel(output_path, index=False)
        
        # Sonucu bildir
        print(f"{interval} verisi kaydedildi: {output_path}")
        print(f"Toplam veri: {len(df)}")
        
        time.sleep(1)
        return True
        
    except Exception as e:
        print(f"{symbol}USDT verisi alınamadı: {str(e)}")
        return False

def collect_all_data():
    # Tüm desteklenen kripto paralar için verileri toplar ve kaydeder.
    # Kripto listesi
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK', 
               'DOT', 'UNI', 'XLM', 'ATOM', 'EOS', 'AAVE', 'IOTA']
    
    # Zaman listesi
    intervals = ['daily', 'monthly', 'hourly']
    
    # Verileri topla
    for interval in intervals:
        print(f"\n{interval} verileri toplanıyor...")
        for symbol in symbols:
            print(f"{symbol} verisi çekiliyor...")
            get_historical_data(symbol, interval)

if __name__ == "__main__":
    collect_all_data()
