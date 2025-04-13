# Emtia Fiyatları ile BIST 100 Yön Tahmini

BIST 100 endeksinin yönünü (artış/azalış) emtia fiyatları, döviz kurları ve finansal göstergeler kullanarak tahmin eden makine öğrenmesi modeli.

## 📋 Proje Hakkında

Bu proje, makine öğrenmesi teknikleriyle BIST 100 endeksinin bir sonraki işlem gününde yükselip yükselmeyeceğini tahmin etmeyi amaçlamaktadır. Tahmin için XGBoost ve Random Forest algoritmaları kullanılmış ve Streamlit ile interaktif bir web arayüzü geliştirilmiştir.

### Veri Kaynakları

- **BIST 100 endeksi (XU100.IS)**
- **Altın Vadeli İşlemleri (GC=F)**
- **Ham Petrol Vadeli İşlemleri (CL=F)**
- **USD/TRY Kuru (USDTRY=X)**
- **ABD 10 Yıllık Tahvil Getirisi (^TNX)**
- **Doğalgaz Vadeli İşlemleri (NG=F)**
- **VIX Volatilite Endeksi (^VIX)**

Tüm veriler Yahoo Finance API'sinden otomatik olarak çekilmektedir.

### Özellikler

- Global piyasa verilerinin detaylı görselleştirmesi
- Korelasyon analizleri (standart ve hareketli korelasyon)
- Lag (gecikme) analizleriyle farklı değişkenlerin BIST 100 üzerindeki gecikmeli etkilerinin tespiti
- XGBoost ile yüksek doğruluklu sınıflandırma modeli
- Model performans metrikleri ve değerlendirme araçları
- Özellik önem analizi (Feature Importance)
- Etkileşimli tahmin göstergeleri ve grafikler

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.7+
- pip veya conda paket yöneticisi

### Kurulum Adımları

1. Projeyi klonlayın:
```bash
git clone https://github.com/yourusername/emtia-bist.git
cd emtia-bist
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## 🏗️ Proje Yapısı

```
emtia-bist/
├── app.py                   # Streamlit web uygulaması ana dosyası
├── data.py                  # Veri işleme ve hazırlama modülü
├── data_collector.py        # Veri toplama ve indirme modülü
├── model.py                 # Model eğitimi, optimizasyon ve değerlendirme modülü
├── model_trainer.py         # Model eğitim süreci ve metadata yönetimi
├── visualization.py         # Veri görselleştirme fonksiyonları
├── requirements.txt         # Proje bağımlılıkları
├── models/                  # Eğitilmiş model ve metadata dosyaları
└── data/                    # Veri dosyaları klasörü
```

### Modüller ve İşlevleri

#### `data_collector.py`
- Yahoo Finance API'sine bağlanarak finansal verileri otomatik olarak çeker
- Veri temizleme ve birleştirme işlemlerini gerçekleştirir
- Çekilen ham verileri `data/` klasörüne kaydeder

#### `data.py`
- Veri önişleme ve özellik mühendisliği işlemlerini yürütür
- Lag (gecikme) özellikleri oluşturur
- Eğitim-test veri ayrımını gerçekleştirir
- Hedef değişkeni (yarınki BIST 100 yönü) oluşturur

#### `model.py`
- Random Forest ve XGBoost modelleri için eğitim fonksiyonları
- Model değerlendirme ve performans ölçümü
- Hiperparametre optimizasyonu ve en iyi modelin seçilmesi
- Model kaydetme ve yükleme işlemleri
- Cross-validation fonksiyonları

#### `model_trainer.py`
- End-to-end model eğitim sürecini otomatikleştirir
- Model metadata yönetimi
- Başarı kriterlerini değerlendirir

#### `visualization.py`
- Veri ve model sonuçlarını görselleştirme fonksiyonları
- Korelasyon analizleri ve ısı haritaları
- Etkileşimli grafikler ve dashboard'lar
- Gecikme (lag) analizleri için görselleştirmeler

#### `app.py`
- Streamlit web uygulaması
- Kullanıcı arayüzü ve sayfa düzeni
- Model tahminleri ve veri keşif arayüzü

## 📊 Kullanım

Uygulama ana olarak 3 sekme içerir:

### 1. Piyasa Verileri
- Farklı zaman aralıklarında piyasa verilerini görüntüleme
- Korelasyon matrisleri ve günlük değişim grafikleri
- Normalize edilmiş fiyat hareketleri

### 2. BIST 100 Tahmini
- En güncel verilerle yarınki BIST 100 yön tahmini
- Tahmin olasılığı ve güven seviyesi
- Modelin önemli bulduğu özelliklerin gösterimi
- Son tahminlerin doğruluk oranları

### 3. Global Değişkenler ve Lag Analizi
- Farklı emtia ve finansal göstergelerin BIST 100 üzerindeki etkilerinin analizi
- Gecikme (lag) analizi ile değişkenlerin gecikmeli etkilerinin tespiti
- Hareketli (rolling) korelasyon analizi
- Global değişkenler dashboard'u

## 🔍 Teknik Detaylar

### Veri İşleme
- Günlük yüzde değişimler temel özellikler olarak kullanılmaktadır
- Farklı gecikme (lag) günleri (1, 10, 30) için özellikler oluşturulmuştur
- NaN değerler forward ve backward filling yöntemleriyle doldurulmuştur

### Model
- XGBoost ve Random Forest sınıflandırıcılar kullanılmıştır
- Hedef değişken: BIST 100'ün bir sonraki gün yönü (1: artış, 0: azalış)
- Modeller 5-katlı çapraz doğrulama ile değerlendirilmiştir
- Doğruluk oranı, F1-skor ve ROC eğrisi ile model performansı ölçülmüştür

## 🤝 Katkıda Bulunma

1. Bu projeyi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## ⚠️ Sorumluluk Reddi

Bu proje eğitim ve araştırma amaçlıdır ve yatırım tavsiyesi niteliği taşımaz. Tüm tahminler gerçek piyasa koşullarında farklılık gösterebilir. Yatırım kararları için profesyonel danışmanlık hizmeti almanız önerilir. 