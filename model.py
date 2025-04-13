"""
BIST 100 Tahmin Projesi - Model Modülü

Bu modül, model eğitimi ve değerlendirme işlemlerini gerçekleştirir.
"""

import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb

def train_model(X_train, y_train, n_estimators=100, random_state=42, model_type="xgboost", **kwargs):
    """
    Model eğitir.
    
    Args:
        X_train: Eğitim seti feature'ları
        y_train: Eğitim seti hedef değişkeni
        n_estimators: Ağaç sayısı
        random_state: Random seed
        model_type: Model tipi ("xgboost")
        **kwargs: Diğer model parametreleri
        
    Returns:
        Eğitilmiş model
    """

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            **kwargs
        )
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Modeli değerlendirir ve performans metriklerini döndürür.
    
    Args:
        model: Eğitilmiş model
        X_test: Test seti feature'ları
        y_test: Test seti hedef değişkeni
        feature_names: Feature isim listesi
        
    Returns:
        dict: Performans metrikleri ve sonuçlar
    """
    # Tahmin yap
    y_pred = model.predict(X_test)
    
    # Performans metriklerini hesapla
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    if feature_names is not None:
        importance = pd.Series(model.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=False)
    else:
        importance = pd.Series(model.feature_importances_)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importance': importance
    }
    
    return results

def cross_validate(model, X, y, cv=5):
    """
    Cross-validation uygular.
    
    Args:
        model: Model
        X: Tüm feature'lar
        y: Tüm hedef değerler
        cv: Fold sayısı
        
    Returns:
        dict: Cross-validation sonuçları
    """
    cv_scores = cross_val_score(model, X, y, cv=cv)
    return {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'cv_scores': cv_scores
    }

def save_model(model, filepath="models/current_model.pkl", create_dir=True):
    """
    Modeli pickle formatında kaydeder.
    
    Args:
        model: Kaydedilecek model
        filepath: Kayıt yolu (varsayılan: models/current_model.pkl)
        create_dir: True ise dizin yoksa oluşturulur
    """
    # Dizin kontrolü ve oluşturma
    directory = os.path.dirname(filepath)
    if create_dir and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath="models/current_model.pkl"):
    """
    Kaydedilmiş modeli yükler.
    
    Args:
        filepath: Model dosya yolu (varsayılan: models/current_model.pkl)
        
    Returns:
        Model: Yüklenen model, dosya bulunamazsa None
    """
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def train_with_different_params(data_file=None, verbose=True):
    """
    Farklı hiperparametrelerle XGBoost modelleri eğitir ve en iyi modeli seçer.
    
    Args:
        data_file: Veri dosyasının yolu
        verbose: İşlem detaylarının yazdırılıp yazdırılmayacağı
        
    Returns:
        tuple: (best_model, best_params, best_accuracy, best_results)
    """
    if verbose:
        print("Farklı parametrelerle XGBoost model eğitimi başlatılıyor...")
    
    # Veriyi yükle
    if data_file is None:
        data_file = os.path.join('data', 'bist_emtia_prepared_data.csv')
    
    # Model trainer'dan load_prepared_data'yı çağırmak yerine burada basit bir veri yükleme işlemi yapalım
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        y = df['target']
        X = df.drop('target', axis=1)
        feature_names = X.columns
    except Exception as e:
        if verbose:
            print(f"Veri yüklenirken hata oluştu: {e}")
        return None, None, 0, None
    
    # Test ve train setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Farklı parametre kombinasyonları
    param_combinations = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
        {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4},
        {'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 7}
    ]
    
    best_accuracy = 0
    best_params = None
    best_model = None
    best_results = None
    
    # Her bir parametre kombinasyonunu dene
    for i, params in enumerate(param_combinations):
        if verbose:
            print(f"\nDeneme {i+1}/{len(param_combinations)}: {params}")
        
        # XGBoost modelini eğit
        model = train_model(
            X_train, 
            y_train, 
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=42,
            model_type="xgboost"
        )
        
        # Modeli değerlendir
        results = evaluate_model(model, X_test, y_test, feature_names)
        accuracy = results['accuracy']
        
        if verbose:
            print(f"Doğruluk oranı: {accuracy:.4f}")
        
        # Daha iyi bir model bulunduysa kaydet
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model
            best_results = results
    
    # En iyi modeli kaydet
    if best_model is not None and verbose:
        print(f"\nEn iyi model (Doğruluk: {best_accuracy:.4f}):")
        print(f"Parametreler: {best_params}")
        
        # Modeli kaydet
        model_path = os.path.join('models', 'bist_model.pkl')
        save_model(best_model, model_path)
        
        # Metadata için cross-validation
        cv_results = cross_validate(best_model, X, y, cv=5)
        
        print(f"Cross-validation doğruluk: {cv_results['mean_cv_score']:.4f}")
        
        # Önemli özellikleri göster
        print("\nÖnemli Özellikler:")
        for feature, importance in best_results['feature_importance'].head(10).items():
            print(f"- {feature}: {importance:.4f}")
        
        # Başarı kriteri kontrolü (0.55'ten büyük doğruluk)
        success = best_accuracy > 0.55
        if success:
            print("\nModel başarı kriterini karşıladı ve kaydedildi!")
        else:
            print("\nModel kaydedildi, ancak başarı kriterini karşılamadı.")
    elif verbose:
        print("Hiçbir model eğitilemedi.")
    
    if verbose:
        print("\nEğitim tamamlandı!")
        
    return best_model, best_params, best_accuracy, best_results
