"""
BIST 100 Tahmin Projesi - Veri İşleme Modülü

Bu modül, BIST 100 ve ilgili emtia verilerinin hazırlanmasıyla ilgili fonksiyonları içerir.
"""

import pandas as pd
import os
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import data_collector

def get_data(start_date, end_date):
    """
    Belirtilen tarih aralığı için BIST 100 ve ilgili emtia verilerini çeker.
    Eğer veri daha önce indirilmişse, yerel dosyadan okur.
    Veri yoksa data_collector.py modülünden collect_data fonksiyonunu çağırır.

    Parameters:
    -----------
    start_date: str
        Veri başlangıç tarihi ('YYYY-MM-DD' formatında)
    end_date: str
        Veri bitiş tarihi ('YYYY-MM-DD' formatında)

    Returns:
    --------
    pd.DataFrame
        İndeksi tarih olan, BIST100, Gold, Oil, USDTRY ve diğer değişkenlerin kapanış değerlerini içeren DataFrame
    """
    csv_path = os.path.join('data', 'bist_emtia_data.csv')
    
    # Eğer veri zaten varsa ve dosya oluşturulmuşsa, dosyadan oku
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Tarih filtrelemesi yap
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
    
    # Veri yoksa data_collector.py modülünden collect_data fonksiyonunu çağır
    try:
        # Veri çekme işlemini gerçekleştir
        # Tarih dönüşümü yap
        if isinstance(end_date, str):
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date_dt = end_date
            
        if isinstance(start_date, str):
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_date_dt = start_date
            
        # Tarihler arasındaki yıl hesabı
        years = (end_date_dt - start_date_dt).days / 365
        years = max(years, 1)  # En az 1 yıllık veri çek
        
        # Veriyi çek
        df, _, _ = data_collector.collect_data(years=years, save=True, verbose=True)
        
        if df is not None:
            # Tarih filtrelemesi yap
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            return df
        else:
            raise Exception("Veri çekme işlemi başarısız oldu.")
    except Exception as e:
        # Manuel veri çekme yöntemine geri dön
        
        # data_collector'dan sembolleri al
        symbols = data_collector.get_symbols()
        
        # Veri çekme
        try:
            # data_collector fonksiyonlarını kullanmaya çalış
            df = data_collector.download_data(start_date, end_date, symbols=symbols, verbose=True)
            if df is not None:
                # Veriyi kaydet
                data_collector.save_raw_data(df, verbose=True)
                return df
            else:
                raise Exception("Veri çekme işlemi başarısız oldu.")
        except Exception as e:
            # Veri çekme
            data_frames = {}
            for name, symbol in symbols.items():
                try:
                    ticker_data = yf.download(
                        symbol, 
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not ticker_data.empty:
                        # Sadece kapanış fiyatlarını al
                        data_frames[name] = ticker_data['Close']
                except Exception as e:
                    pass
            
            # Verileri tek bir DataFrame'de birleştir
            df = pd.DataFrame()
            for name, series in data_frames.items():
                df[name] = series
            
            # NaN değerleri doldur (varsa)
            if df.isna().any().any():
                df = df.ffill()  # Forward fill
                df = df.bfill()  # Backward fill - kalan NaN değerler için
            
            # Klasörü oluştur
            os.makedirs('data', exist_ok=True)
            
            # Veriyi kaydet
            df.to_csv(csv_path)
            
            return df

def add_lag_features(df_pct, lag_days=[1, 10, 30]):
    """
    Verilen günlük değişim DataFrame'i için gecikmeli (lag) özellikler ekler.
    
    Parameters:
    -----------
    df_pct: pd.DataFrame
        Günlük değişim oranlarını içeren DataFrame
    lag_days: list, default=[1, 10, 30]
        Gecikmeli gün sayıları listesi
        
    Returns:
    --------
    pd.DataFrame
        Gecikmeli özelliklerin eklendiği DataFrame
    """
    df_with_lags = df_pct.copy()
    
    # BIST100 dışındaki sütunlar için lag ekle
    external_cols = [col for col in df_pct.columns if not col.startswith('BIST100')]
    
    for col in external_cols:
        for lag in lag_days:
            # Gecikmeli değerleri ekle
            lag_col_name = f"{col}_lag{lag}"
            df_with_lags[lag_col_name] = df_pct[col].shift(lag)
    
    # NaN değerleri temizle
    df_with_lags = df_with_lags.dropna()
    
    return df_with_lags

def prepare_data(df, use_lags=True, lag_days=[1, 10, 30]):
    """
    Ham veriyi model için hazırlar. Feature'ları oluşturur ve ölçeklendirir.

    Parameters:
    -----------
    df: pd.DataFrame
        BIST100 ve diğer finansal değişkenlerin kapanış değerlerini içeren DataFrame
    use_lags: bool, default=True
        Gecikmeli özelliklerin eklenip eklenmeyeceği
    lag_days: list, default=[1, 10, 30]
        Eğer use_lags=True ise, eklenecek gecikmeli gün sayıları

    Returns:
    --------
    X: np.ndarray
        Ölçeklendirilmiş özellikler
    y: np.ndarray
        Hedef değişken (1: yükseliş, 0: düşüş)
    scaler: StandardScaler
        Verileri ölçeklendiren scaler
    feature_names: list
        Feature isimlerinin listesi
    """
    # Günlük değişim oranlarını hesapla
    df_pct = df.pct_change().dropna()
    df_pct.columns = [f"{col}_change" for col in df_pct.columns]
    
    # Gecikmeli özellikleri ekle
    if use_lags:
        df_pct = add_lag_features(df_pct, lag_days)
    
    # Hedef değişken oluştur (BIST100 yarın yükselecek mi?)
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
    
    return X_scaled, y_data.values, scaler, X_data.columns.tolist()

def analyze_lag_importance(df, target_col='BIST100'):
    """
    Farklı emtia ve finansal göstergelerin BIST100 üzerindeki gecikmeli 
    etkilerini korelasyon analizi ile değerlendirir.
    
    Parameters:
    -----------
    df: pd.DataFrame
        Ham finansal verileri içeren DataFrame
    target_col: str, default='BIST100'
        Analiz edilecek hedef sütun
        
    Returns:
    --------
    pd.DataFrame
        Her değişken ve gecikme için korelasyon değerlerini içeren DataFrame
    """
    # Günlük değişimleri hesapla
    df_pct = df.pct_change().dropna()
    
    # Hedef sütun
    target = df_pct[target_col]
    
    # Diğer sütunlar
    other_cols = [col for col in df_pct.columns if col != target_col]
    
    # Sonuçları saklamak için DataFrame
    lag_corrs = []
    
    # Farklı lag değerleri için korelasyonları hesapla
    max_lag = 30
    for col in other_cols:
        for lag in range(1, max_lag + 1):
            # Gecikmeli seri
            lagged_series = df_pct[col].shift(lag)
            
            # Geçerli veri aralığı
            valid_idx = lagged_series.dropna().index.intersection(target.index)
            
            if len(valid_idx) > 0:
                # Korelasyon
                corr = target.loc[valid_idx].corr(lagged_series.loc[valid_idx])
                
                lag_corrs.append({
                    'variable': col,
                    'lag_days': lag,
                    'correlation': corr
                })
    
    # Sonuçları DataFrame'e dönüştür
    lag_corr_df = pd.DataFrame(lag_corrs)
    
    # Değişken ve gecikme günü bazında pivot tablo oluştur
    lag_corr_pivot = lag_corr_df.pivot(index='variable', columns='lag_days', values='correlation')
    
    return lag_corr_df, lag_corr_pivot

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test kümelerine ayırır.

    Parameters:
    -----------
    X: np.ndarray
        Özellikler
    y: np.ndarray
        Hedef değişken
    test_size: float, default=0.2
        Test kümesinin toplam veriye oranı
    random_state: int, default=42
        Rastgele ayırma için seed değeri

    Returns:
    --------
    X_train: np.ndarray
        Eğitim kümesi özellikleri
    X_test: np.ndarray
        Test kümesi özellikleri
    y_train: np.ndarray
        Eğitim kümesi hedef değişkeni
    y_test: np.ndarray
        Test kümesi hedef değişkeni
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 