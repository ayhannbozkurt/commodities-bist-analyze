"""
BIST 100 Tahmin Projesi - Model Eğitimi Modülü

Bu modül, hazırlanmış verileri kullanarak model eğitimi, 
değerlendirme ve kaydetme fonksiyonlarını içerir.
"""

import os
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split

# model.py modülünden fonksiyonları import et
from model import (
    train_model, 
    evaluate_model, 
    cross_validate, 
    save_model, 
    load_model
)

def load_prepared_data(data_file=None):
    """
    Hazırlanmış veriyi yükler.
    
    Parameters:
    -----------
    data_file: str, optional
        Veri dosyasının yolu. None ise varsayılan konum kullanılır.
        
    Returns:
    --------
    tuple
        (X, y, feature_names) - öznitelikler, hedef değişken ve öznitelik adları
    """
    if data_file is None:
        data_file = os.path.join('data', 'bist_emtia_prepared_data.csv')
    
    if not os.path.exists(data_file):
        return None, None, None
    
    # Veriyi yükle
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # X ve y ayırma
    y = df['target']
    X = df.drop('target', axis=1)
    feature_names = X.columns
    
    return X, y, feature_names

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test setlerine ayırır.
    
    Parameters:
    -----------
    X: pd.DataFrame
        Öznitelikler
    y: pd.Series
        Hedef değişken
    test_size: float, optional
        Test seti oranı
    random_state: int, optional
        Rastgelelik için seed değeri
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - eğitim ve test setleri
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def save_model_metadata(model, results, cv_results, X_train, X_test, feature_names, model_path, model_type):
    """
    Model metadata'sını kaydeder.
    
    Parameters:
    -----------
    model: sklearn model
        Eğitilmiş model
    results: dict
        Model değerlendirme sonuçları
    cv_results: dict
        Cross-validation sonuçları
    X_train: pd.DataFrame
        Eğitim veri seti
    X_test: pd.DataFrame
        Test veri seti
    feature_names: pd.Index
        Öznitelik adları
    model_path: str
        Model dosyasının kaydedildiği yol
    model_type: str
        Kullanılan model tipi
        
    Returns:
    --------
    str
        Metadata dosyasının yolu
    """
    # Model meta verilerini oluştur
    metadata = {
        'feature_names': feature_names.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'accuracy': results['accuracy'],
        'cv_score': cv_results['mean_cv_score'],
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': model_type,
        'model_parameters': {
            'n_estimators': model.n_estimators,
            'random_state': model.random_state if hasattr(model, 'random_state') else None
        }
    }
    
    # Metadata'yı, model ile aynı dizine kaydet
    metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.pkl')
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    return metadata_path

def check_success_criteria(results, threshold=0.55):
    """
    Modelin başarı kriterini karşılayıp karşılamadığını kontrol eder.
    
    Parameters:
    -----------
    results: dict
        Model değerlendirme sonuçları
    threshold: float, optional
        Başarı kriteri eşik değeri
        
    Returns:
    --------
    bool
        Başarı kriteri karşılandıysa True, aksi halde False
    """
    accuracy = results['accuracy']
    
    return accuracy > threshold

def train_and_evaluate_model(data_file=None, test_size=0.2, n_estimators=100, 
                           random_state=42, model_name="current_model", model_type="xgboost"):
    """
    Veri yükleme, model eğitimi, değerlendirme ve kaydetme işlemlerini birleştirir.
    
    Parameters:
    -----------
    data_file: str, optional
        Veri dosyasının yolu
    test_size: float, optional
        Test seti oranı
    n_estimators: int, optional
        Ağaç sayısı
    random_state: int, optional
        Rastgelelik için seed değeri
    model_name: str, optional
        Kaydedilecek model dosyasının adı
    model_type: str, optional
        Kullanılacak model tipi ("xgboost")
        
    Returns:
    --------
    tuple
        (model, results, cv_results, model_path, metadata_path)
    """
    # Veriyi yükle
    X, y, feature_names = load_prepared_data(data_file)
    if X is None:
        return None, None, None, None, None
    
    # Veriyi ayır
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Modeli eğit
    model = train_model(X_train, y_train, n_estimators, random_state, model_type)
    
    # Modeli değerlendir
    results = evaluate_model(model, X_test, y_test, feature_names)
    
    # Cross-validation uygula
    cv_results = cross_validate(model, X, y, cv=5)
    
    # Modeli kaydet - varsayılan olarak models/current_model.pkl olacak
    model_path = os.path.join('models', f"{model_name}.pkl")
    save_model(model, model_path)
    
    # Model metadata'sını kaydet
    metadata_path = save_model_metadata(
        model, results, cv_results, X_train, X_test, 
        feature_names, model_path, model_type
    )
    
    # Başarı kriterini kontrol et
    success = check_success_criteria(results)
    
    return model, results, cv_results, model_path, metadata_path, success

if __name__ == "__main__":
    # Ana işlevi çağır
    model, results, cv_results, model_path, metadata_path, success = train_and_evaluate_model() 