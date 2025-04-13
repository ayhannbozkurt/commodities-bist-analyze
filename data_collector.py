"""
BIST 100 Tahmin Projesi - Veri Toplama Script

Bu script, Yahoo Finance'den gerçek veri çeker ve data/ klasörüne kaydeder.
"""

import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle

def create_directories():
    """Gerekli dizinleri oluşturur"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def get_symbols():
    """Çekilecek sembolleri döndürür"""
    return {
        "BIST100": "XU100.IS",  # BIST 100 endeksi (Istanbul Stock Exchange)
        "Gold": "GC=F",         # Altın vadeli işlemler (Gold Futures)
        "Oil": "CL=F",          # Ham petrol vadeli işlemler (Crude Oil Futures)
        "USDTRY": "USDTRY=X",   # USD/TRY kuru (US Dollar - Turkish Lira)
        "US10Y": "^TNX",        # ABD 10 Yıllık Tahvil Getirisi
        "NatGas": "NG=F",       # Doğal Gaz vadeli işlemler
        "VIX": "^VIX"           # Volatilite Endeksi (Korku Endeksi)
    }

def download_data(start_date, end_date, symbols=None, verbose=True):
    """
    Belirtilen sembollerin verilerini Yahoo Finance'den çeker
    
    Parameters:
    -----------
    start_date: datetime veya str
        Başlangıç tarihi
    end_date: datetime veya str
        Bitiş tarihi
    symbols: dict, optional
        İndirilecek sembollerin listesi. None ise get_symbols() kullanılır
    verbose: bool, default=True
        İşlem detaylarının ekrana yazdırılması
        
    Returns:
    --------
    pd.DataFrame
        Çekilen verileri içeren DataFrame
    """
    # Tarihleri string formatına dönüştür
    if isinstance(start_date, datetime):
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date_str = start_date
        
    if isinstance(end_date, datetime):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = end_date
    
    # Semboller belirlenmemişse varsayılanları kullan
    if symbols is None:
        symbols = get_symbols()
    
    # Veri çekme
    data_frames = {}
    for name, symbol in symbols.items():
        try:
            ticker_data = yf.download(
                symbol, 
                start=start_date_str, 
                end=end_date_str,
                progress=False
            )
            
            if not ticker_data.empty:
                # Sadece kapanış fiyatlarını al
                data_frames[name] = ticker_data['Close']
        except Exception as e:
            pass

    # Verileri tek bir DataFrame'de birleştir
    if len(data_frames) == 0:
        return None

    # Tüm sembolleri tek bir DataFrame'e dönüştür
    df = pd.DataFrame()
    for name, series in data_frames.items():
        df[name] = series

    # NaN değerleri doldur (varsa)
    if df.isna().any().any():
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill - kalan NaN değerler için

    return df

def save_raw_data(df, output_path=None, verbose=True):
    """
    Ham veriyi CSV dosyasına kaydeder
    
    Parameters:
    -----------
    df: pd.DataFrame
        Kaydedilecek ham veri
    output_path: str, optional
        Kaydedilecek dosya yolu. None ise varsayılan konum kullanılır
    verbose: bool, default=True
        İşlem detaylarının ekrana yazdırılması
        
    Returns:
    --------
    str
        Kaydedilen dosyanın yolu
    """
    if output_path is None:
        create_directories()
        output_path = os.path.join('data', 'bist_emtia_data.csv')
        
    df.to_csv(output_path)
    
    return output_path

def prepare_training_data(df, verbose=True):
    """
    Ham veriden model eğitimi için gerekli verileri hazırlar
    
    Parameters:
    -----------
    df: pd.DataFrame
        Ham veri
    verbose: bool, default=True
        İşlem detaylarının ekrana yazdırılması
        
    Returns:
    --------
    tuple
        (X_data, y_data, X_scaled_df, prepared_data)
    """
    # Günlük değişim oranlarını hesapla
    df_pct = df.pct_change().dropna()
    df_pct.columns = [f"{col}_change" for col in df_pct.columns]

    # Hedef değişken oluştur
    # Bugünkü BIST100 değeri ile yarınki değer arasındaki ilişki
    df_target = pd.DataFrame(index=df.index[:-1])  # Son günün tahmini olmayacak
    df_target['BIST100_today'] = df['BIST100'][:-1].values
    df_target['BIST100_tomorrow'] = df['BIST100'][1:].values
    df_target['target'] = (df_target['BIST100_tomorrow'] > df_target['BIST100_today']).astype(int)

    # Temiz veri hazırlama - aynı indekslere sahip veriler oluştur
    clean_idx = df_pct.index.intersection(df_target.index)
    X_data = df_pct.loc[clean_idx]
    y_data = df_target.loc[clean_idx, 'target']

    # Feature standardizasyonu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_data.index, columns=X_data.columns)

    # Model için veriyi hazırla
    prepared_data = X_scaled_df.copy()
    prepared_data['target'] = y_data
    
    return X_data, y_data, X_scaled_df, prepared_data

def save_prepared_data(prepared_data, output_path=None, verbose=True):
    """
    Hazırlanmış veriyi CSV dosyasına kaydeder
    
    Parameters:
    -----------
    prepared_data: pd.DataFrame
        Kaydedilecek hazırlanmış veri
    output_path: str, optional
        Kaydedilecek dosya yolu. None ise varsayılan konum kullanılır
    verbose: bool, default=True
        İşlem detaylarının ekrana yazdırılması
        
    Returns:
    --------
    str
        Kaydedilen dosyanın yolu
    """
    if output_path is None:
        create_directories()
        output_path = os.path.join('data', 'bist_emtia_prepared_data.csv')
        
    prepared_data.to_csv(output_path)
    
    return output_path

def save_metadata(df, X_data, y_data, verbose=True):
    """
    Meta verileri bir metin dosyasına kaydeder
    
    Parameters:
    -----------
    df: pd.DataFrame
        Ham veri
    X_data: pd.DataFrame
        Feature'lar
    y_data: pd.Series
        Hedef değişken
    verbose: bool, default=True
        İşlem detaylarının ekrana yazdırılması
        
    Returns:
    --------
    str
        Kaydedilen dosyanın yolu
    """
    metadata_path = os.path.join('data', 'metadata.pkl')
    
    # Meta verileri hesapla
    positive_ratio = y_data.mean()
    
    # Meta verileri sözlük olarak sakla
    metadata = {
        'raw_data_shape': df.shape,
        'feature_count': len(X_data.columns),
        'date_range': (df.index.min(), df.index.max()),
        'positive_ratio': positive_ratio,
        'columns': df.columns.tolist(),
        'feature_columns': X_data.columns.tolist(),
        'data_collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Meta verileri kaydet
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata_path

def collect_data(years=5, save=True, verbose=True):
    """
    Tüm veri toplama ve hazırlama sürecini otomatikleştirir
    
    Parameters:
    -----------
    years: int veya float, default=5
        Kaç yıllık veri çekileceği
    save: bool, default=True
        Verinin kaydedilip kaydedilmeyeceği
    verbose: bool, default=True
        İşlem detaylarının ekrana yazdırılması
        
    Returns:
    --------
    tuple
        (ham_veri, hazırlanmış_veri, meta_veri_yolu)
    """
    # Tarih aralığını belirle
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * years))
    
    # Verileri indir
    df = download_data(start_date, end_date, verbose=verbose)
    if df is None:
        return None, None, None
        
    # Ham veriyi kaydet
    if save:
        raw_data_path = save_raw_data(df, verbose=verbose)
    
    # Veriyi hazırla
    X_data, y_data, X_scaled, prepared_data = prepare_training_data(df, verbose=verbose)
    
    # Hazırlanmış veriyi kaydet
    if save:
        prepared_data_path = save_prepared_data(prepared_data, verbose=verbose)
        metadata_path = save_metadata(df, X_data, y_data, verbose=verbose)
    
    return df, prepared_data, metadata_path

if __name__ == "__main__":
    create_directories()
    df, prepared_data, metadata_path = collect_data(years=5, save=True, verbose=True) 