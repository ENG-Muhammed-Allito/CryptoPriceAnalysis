import pandas as pd
import os
import glob
import numpy as np

def merge_crypto_data(interval):
    """
    Belirli bir zaman aralığı için kripto para verilerini birleştirme
    
    Parameters:
    -----------
    interval : str
        Zaman aralığı ('Daily', 'Hourly', 'Monthly')
    """
    print(f"\n{interval} verileri birleştiriliyor...")
    
    # Klasör yollarını belirleme
    input_dir = f"d:/Crypto_Analysis_random_state/data/{interval}_data"
    output_dir = f"d:/Crypto_Analysis_random_state/Karma_data/{interval}_data"
    
    # Çıktı klasörünün varlığını kontrol etme
    os.makedirs(output_dir, exist_ok=True)
    
    # Excel dosyalarının listesini alma
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    
    if not excel_files:
        print(f"Klasörde dosya bulunamadı: {input_dir}")
        return
    
    # Veri çerçevelerini depolamak için liste
    dfs = []
    
    # Her dosyayı okuma ve symbol sütunu ekleme
    for file in excel_files:
        # Dosya adından kripto para sembolünü çıkarma
        symbol = os.path.basename(file).split('.')[0]
        print(f"  {symbol} işleniyor...")
        
        # Excel dosyasını okuma
        df = pd.read_excel(file)
        
        # Symbol sütunu ekleme
        df['symbol'] = symbol
        
        # Veri çerçevesini listeye ekleme
        dfs.append(df)
    
    # Tüm veri çerçevelerini birleştirme
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Verileri rastgele karıştırma
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Birleştirilmiş verileri kaydetme
    output_file = os.path.join(output_dir, f"all_crypto_{interval.lower()}.xlsx")
    merged_df.to_excel(output_file, index=False)
    
    print(f"Birleştirilmiş veriler kaydedildi: {output_file}")
    print(f"Satır sayısı: {merged_df.shape[0]}, Sütun sayısı: {merged_df.shape[1]}")
    print(f"Birleştirilen kripto para sayısı: {merged_df['symbol'].nunique()}")

def merge_all_intervals():
    """
    Tüm zaman aralıkları için kripto para verilerini birleştirme
    """
    print("Kripto para verilerini birleştirme işlemi başlatılıyor...")
    
    # Her zaman aralığı için verileri birleştirme
    merge_crypto_data("Daily")
    merge_crypto_data("Hourly")
    merge_crypto_data("Monthly")
    
    print("\nBirleştirme işlemi başarıyla tamamlandı!")

if __name__ == "__main__":
    merge_all_intervals()
