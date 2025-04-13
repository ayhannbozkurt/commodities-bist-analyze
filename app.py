"""
BIST 100 Tahmin Projesi - Streamlit App

Bu uygulama, emtia fiyatlarıyla BIST 100 endeksinin yön tahmini için oluşturulan modeli gösterir.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Model modülü
from data import analyze_lag_importance
from visualization import (
    plot_feature_importance, 
    plot_correlation_matrix,
    plot_daily_changes,
    plot_recent_predictions,
    prepare_display_data,
    plot_lag_correlation_heatmap,
    plot_lag_effect_line, 
    plot_rolling_correlation,
    plot_global_variables_dashboard,
    plot_enhanced_correlation_matrix, 
)

# Sayfa yapılandırması
st.set_page_config(
    page_title="BIST 100 Tahmin Uygulaması",
    page_icon="📈",
    layout="wide"
)

# Başlık
st.title("📊 Emtia Fiyatları ile BIST 100 Yön Tahmini")
st.markdown("---")

# Hazır veri ve modeli yükle
@st.cache_data
def load_prepared_data():
    """Hazır veriyi yükle"""
    try:
        df = pd.read_csv('data/bist_emtia_data.csv', index_col=0, parse_dates=True)
        prepared_df = pd.read_csv('data/bist_emtia_prepared_data.csv', index_col=0, parse_dates=True)
        
        # Meta verileri yükle
        with open('data/metadata.txt', 'r') as f:
            metadata = {}
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()
        
        return df, prepared_df, metadata
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None, None, None

@st.cache_resource
def load_trained_model():
    """Eğitilmiş modeli yükle"""
    model_path = 'models/bist_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {e}")
    return None

# Lag analizi hesaplama
@st.cache_data
def get_lag_analysis(df):
    """Lag analizi yap ve sonuçları döndür"""
    try:
        lag_corr_df, lag_corr_pivot = analyze_lag_importance(df)
        return lag_corr_df, lag_corr_pivot
    except Exception as e:
        st.error(f"Lag analizi yapılırken hata oluştu: {e}")
        return None, None

# Veri ve modeli yükle
with st.spinner("Veriler ve model yükleniyor..."):
    raw_data, prepared_data, metadata = load_prepared_data()
    model = load_trained_model()
    
    # Lag analizi sonuçlarını hesapla
    if raw_data is not None:
        lag_corr_df, lag_corr_pivot = get_lag_analysis(raw_data)

# Ana içerik
if raw_data is not None and not raw_data.empty:
    # Tab'lar oluştur - Yeni global değişkenler ve lag analizi sekmesi eklendi
    tab1, tab2, tab3 = st.tabs(["📈 Piyasa Verileri", "🔮 BIST 100 Tahmini", "🌍 Global Değişkenler ve Lag Analizi"])
    
    # Tab 1: Piyasa Verileri
    with tab1:
        st.header("Piyasa Verileri")
        
        # Veri aralığını göster
        st.info(f"Veri aralığı: {raw_data.index.min().strftime('%d-%m-%Y')} - {raw_data.index.max().strftime('%d-%m-%Y')}")
        
        # Görselleştirme için zaman aralığı seçimi
        col1, col2 = st.columns(2)
        with col1:
            view_period = st.radio(
                "Görüntüleme periyodu:",
                ["Son 1 Ay", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "Tümü"],
                index=1
            )
        
        with col2:
            normalize = st.checkbox("Grafikleri normalize et", value=False)
        
        # Veriyi hazırla
        plot_data, data_view = prepare_display_data(raw_data, view_period, normalize)

        # Gelişmiş Korelasyon Matrisi (YENİ)
        st.subheader("BIST100 ile Emtialar Arasındaki Korelasyon")
        fig = plot_enhanced_correlation_matrix(data_view, f"{view_period} - BIST100 ve Emtialar/Göstergeler Arasındaki Korelasyon")
        st.plotly_chart(fig, use_container_width=True)
        
        # Korelasyon matrisi
        st.subheader("Korelasyon Matrisi")
        fig = plot_correlation_matrix(data_view, view_period)
        st.plotly_chart(fig, use_container_width=True)
        
        # Günlük değişim (volatilite) grafiği
        st.subheader("Günlük Değişim Oranları")
        fig = plot_daily_changes(data_view, view_period)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: BIST 100 Tahmini
    with tab2:
        st.header("BIST 100 Yön Tahmini")
        
        if model is not None:
            # Son verileri al
            latest_data = raw_data.iloc[-1:]
            
            # Son 5 günün fiyatlarını göster
            st.subheader("Son 5 Gün Fiyat Değişimleri")
            
            last_5_days = raw_data.tail(5)
            
            # Fiyat tablosu
            st.dataframe(last_5_days.style.format("{:.2f}"))
            
            # Günlük değişimler
            daily_change = last_5_days.pct_change() * 100
            daily_change = daily_change.iloc[1:].copy()  # İlk NaN satırını kaldır
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                last_change = daily_change['BIST100'].iloc[-1]
                delta_color = "normal" if last_change >= 0 else "inverse"
                st.metric("BIST 100", f"{raw_data['BIST100'].iloc[-1]:.2f}", f"{last_change:.2f}%", delta_color=delta_color)
            
            with col2:
                last_change = daily_change['Gold'].iloc[-1]
                delta_color = "normal" if last_change >= 0 else "inverse"
                st.metric("Altın", f"{raw_data['Gold'].iloc[-1]:.2f}", f"{last_change:.2f}%", delta_color=delta_color)
            
            with col3:
                last_change = daily_change['Oil'].iloc[-1]
                delta_color = "normal" if last_change >= 0 else "inverse"
                st.metric("Petrol", f"{raw_data['Oil'].iloc[-1]:.2f}", f"{last_change:.2f}%", delta_color=delta_color)
            
            with col4:
                last_change = daily_change['USDTRY'].iloc[-1]
                delta_color = "inverse" if last_change >= 0 else "normal"  # USD/TRY için ters mantık
                st.metric("USD/TRY", f"{raw_data['USDTRY'].iloc[-1]:.2f}", f"{last_change:.2f}%", delta_color=delta_color)
            
            # Model için input feature'ları hazırla
            # Modelin beklediği doğru sütunları belirle
            model_metadata_path = os.path.join('models', 'model_metadata.pkl')
            model_features = None
            if os.path.exists(model_metadata_path):
                with open(model_metadata_path, 'rb') as f:
                    model_metadata = pickle.load(f)
                    if 'feature_names' in model_metadata:
                        model_features = model_metadata['feature_names']

            # Model features bulundu mu kontrol et ve ona göre feature'ları seç
            if model_features:
                # Tüm değişkenler için günlük değişimi hesapla
                all_change_features = daily_change.iloc[-1:].copy()
                
                # Model için gerekli olan sütunları seç
                if all(feat in all_change_features.columns for feat in model_features):
                    features = all_change_features[model_features].values.reshape(1, -1)
                    st.info(f"Model {len(model_features)} özellik kullanıyor: {', '.join(model_features)}")
                else:
                    # Eğer model_features'daki bazı sütunlar mevcut değilse
                    missing_features = [feat for feat in model_features if feat not in all_change_features.columns]
                    
                    # Mevcut olan özellikleri kullan ve eksik olanlar için 0 ata
                    features = np.zeros((1, len(model_features)))
                    for i, feat in enumerate(model_features):
                        if feat in all_change_features.columns:
                            features[0, i] = all_change_features[feat].values[0]
                    
            else:
                # Model features bulunamadıysa, sadece mevcut değişimleri kullan
                features = daily_change.iloc[-1:].values.reshape(1, -1)
                st.warning("Model metadata bulunamadı. Mevcut tüm değişkenler kullanılıyor.")
            
            # Tahmin yap
            try:
                prediction = model.predict(features)[0]
                probability = float(model.predict_proba(features)[0][1])
                
                st.subheader("Yarınki BIST 100 Tahmini")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("📈 BIST 100'ün yarın **YÜKSELMESI** bekleniyor!")
                    else:
                        st.error("📉 BIST 100'ün yarın **DÜŞMESI** bekleniyor!")
                
                with col2:
                    st.metric("Yükselme Olasılığı", f"{probability:.2%}")
                    st.progress(probability)
                
                st.caption(f"Son verilerle tahmin yapıldı. Tarih: {latest_data.index[0].strftime('%Y-%m-%d')}")
                
                # Model performans bilgisi
                model_metadata_path = os.path.join('models', 'model_metadata.pkl')
                if os.path.exists(model_metadata_path):
                    with open(model_metadata_path, 'rb') as f:
                        model_metadata = pickle.load(f)
                    
                    # Feature importance'ı göster
                    if "feature_importance" in model_metadata:
                        st.subheader("Tahmin İçin Önemli Faktörler")
                        
                        feature_importance = pd.Series(
                            model.feature_importances_,
                            index=model_metadata.get('feature_names', [])
                        ).sort_values(ascending=False)
                        
                        fig = plot_feature_importance(feature_importance, title="Özellik Önem Derecesi")
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Tahmin yapılırken hata oluştu: {e}")
                st.info("Tahmin modelinin beklediği özellikler ile mevcut verilerin uyumlu olmama ihtimali var.")
            
            # Geçmiş tahmin performansı
            st.subheader("Geçmiş Tahmin Performansı")
            
            # Son 30 günün tahminleri
            if prepared_data is not None and 'target' in prepared_data.columns:
                fig = plot_recent_predictions(prepared_data, n_days=30)
                st.plotly_chart(fig, use_container_width=True)
                
                # İstatistikler
                last_30_predictions = prepared_data.tail(30)
                up_days = last_30_predictions['target'].sum()
                down_days = len(last_30_predictions) - up_days
                
                st.info(f"Son 30 gün içinde BIST 100 {up_days} gün yükseldi, {down_days} gün düştü.")
                
    
    # Tab 3: Global Değişkenler ve Lag Analizi
    with tab3:
        st.header("Global Değişkenler ve Lag Analizi")
        
        # Alt tablar oluştur
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Global Değişkenler", "🕒 Gecikmeli Etkiler", "📉 Hareketli Korelasyonlar"])
        
        # Global Değişkenler
        with sub_tab1:
            st.subheader("Global Değişkenler Analiz Dashboardu")
            
            # Dönem seçimi
            period = st.radio(
                "Görüntüleme periyodu:",
                ["1M", "3M", "6M", "1Y", "Tümü"],
                index=2,
                horizontal=True
            )
            
            # Dashboard görseli
            fig = plot_global_variables_dashboard(raw_data, period)
            st.plotly_chart(fig, use_container_width=True)
        
        # Gecikmeli Etkiler
        with sub_tab2:
            st.subheader("Gecikmeli Etki Analizi")
            
            st.info("""
            **Gecikmeli Etki Analizi Nedir?**
            
            Bu analiz, farklı global değişkenlerin BIST 100 üzerindeki etkisinin ne kadar süre sonra 
            gerçekleştiğini gösterir. Örneğin, petrol fiyatlarındaki bir değişimin BIST 100 üzerindeki 
            etkisi hemen mi yoksa birkaç gün sonra mı ortaya çıkıyor?
            """)
            
            if lag_corr_df is not None and lag_corr_pivot is not None:
                # Isı haritası
                st.subheader("Gecikmeli Etki Korelasyon Isı Haritası")
                fig = plot_lag_correlation_heatmap(lag_corr_pivot)
                st.plotly_chart(fig, use_container_width=True)
                
                # Çizgi grafiği
                st.subheader("Değişkenlerin Gecikmeli Etkileri")
                
                # Değişken seçimi
                all_variables = lag_corr_df['variable'].unique().tolist()
                selected_vars = st.multiselect(
                    "Görselleştirilecek değişkenleri seçin:",
                    all_variables,
                    default=all_variables[:3],  # İlk 3 değişkeni default olarak seç
                    key="lag_analysis_vars"
                )
                
                if selected_vars:
                    fig = plot_lag_effect_line(lag_corr_df, selected_vars)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"""
                    **Yorumlama:** 
                    - Pozitif korelasyon, değişkenin BIST 100 ile aynı yönde hareket ettiğini gösterir
                    - Negatif korelasyon, değişkenin BIST 100 ile ters yönde hareket ettiğini gösterir
                    - Korelasyonun en yüksek olduğu gecikme günü, değişkenin BIST 100 üzerindeki etkisinin 
                      en güçlü hissedildiği zaman aralığını gösterir
                    """)
                else:
                    st.warning("Lütfen en az bir değişken seçin.")
            else:
                st.error("Lag analizi sonuçları hesaplanamadı.")
        
        # Hareketli Korelasyonlar
        with sub_tab3:
            st.subheader("Zaman İçinde Değişen Korelasyonlar")
            
            st.info("""
            **Hareketli Korelasyon Analizi Nedir?**
            
            Bu analiz, global değişkenler ile BIST 100 arasındaki ilişkinin zaman içinde nasıl değiştiğini gösterir.
            Bazı dönemlerde bir değişkenin etkisi artarken, başka dönemlerde azalabilir.
            """)
            
            # Pencere büyüklüğü seçimi
            window_size = st.slider(
                "Analiz pencere büyüklüğü (gün):",
                min_value=30,
                max_value=180,
                value=90,
                step=30
            )
            
            # Değişken seçimi
            variables = [col for col in raw_data.columns if col != "BIST100"]
            selected_vars = st.multiselect(
                "Görselleştirilecek değişkenleri seçin:",
                variables,
                default=variables[:3],  # İlk 3 değişkeni default olarak seç
                key="rolling_corr_vars"
            )
            
            if selected_vars:
                fig = plot_rolling_correlation(raw_data, "BIST100", window_size, selected_vars)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"""
                **Yorumlama:** 
                - Korelasyonun artması, ilgili değişkenin BIST 100 üzerindeki etkisinin güçlendiğini gösterir
                - Korelasyonun işaret değiştirmesi, ilişkinin yönünün değiştiğini gösterir
                - Korelasyon sıfıra yaklaştığında, değişkenin etkisi azalıyor demektir
                """)
            else:
                st.warning("Lütfen en az bir değişken seçin.")
            
else:
    st.error("Veriler yüklenemedi. Lütfen önce veri toplama işlemini gerçekleştirin.")
    st.info("Veri toplamak için 'python data_collector.py' komutunu çalıştırın.")

# Bilgi
st.markdown("---")
st.info("""
**📌 Not:** Bu uygulama eğitim amacıyla oluşturulmuştur ve gerçek yatırım tavsiyesi içermez. 
Tahminler geçmiş veriler üzerinde eğitilmiş bir model tarafından yapılmakta olup, 
gelecekteki gerçek performansı garanti etmez.
""")

# Footer
st.markdown("---")
st.caption("BIST 100 Tahmin Projesi © 2025") 