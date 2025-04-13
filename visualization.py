"""
BIST 100 Tahmin Projesi - Görselleştirme Modülü

Bu modül, model sonuçlarının ve verilerin görselleştirilmesi işlemlerini gerçekleştirir.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_feature_importance(importance, title="Feature Importance"):
    """
    Feature importance grafiği oluşturur.
    
    Args:
        importance: Feature importance Series/DataFrame
        title: Grafik başlığı
    
    Returns:
        Plotly figure
    """
    fig = px.bar(
        importance.reset_index(),
        x="index",
        y=0,
        title=title,
        labels={"index": "Özellikler", "0": "Önem"}
    )
    return fig

def plot_correlation_matrix(data_view, view_period):
    """
    Korelasyon matrisi grafiği oluşturur.
    
    Args:
        data_view: Görselleştirilecek veri DataFrame'i
        view_period: Görüntüleme periyodu (string)
        
    Returns:
        Plotly figure
    """
    corr = data_view.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title=f"{view_period} Korelasyon Matrisi"
    )
    fig.update_layout(height=500)
    
    return fig

def plot_daily_changes(data_view, view_period):
    """
    Günlük değişim oranlarının grafiğini oluşturur.
    
    Args:
        data_view: Görselleştirilecek veri DataFrame'i
        view_period: Görüntüleme periyodu (string)
        
    Returns:
        Plotly figure
    """
    daily_change = data_view.pct_change().dropna() * 100
    
    fig = go.Figure()
    
    for col in daily_change.columns:
        fig.add_trace(go.Scatter(
            x=daily_change.index,
            y=daily_change[col],
            mode='lines',
            name=col
        ))
        
    fig.update_layout(
        title=f"{view_period} Günlük Değişim (%)",
        xaxis_title="Tarih",
        yaxis_title="Günlük Değişim (%)",
        legend_title="Sembol",
        height=500
    )
    
    return fig

def plot_recent_predictions(prediction_data, n_days=30):
    """
    Son n gün için tahmin performansını gösteren grafik.
    
    Args:
        prediction_data: Tahmin sonuçlarını içeren DataFrame
        n_days: Gösterilecek gün sayısı
        
    Returns:
        Plotly figure
    """
    last_n_predictions = prediction_data.tail(n_days).copy()
    
    fig = go.Figure()
    
    # Gerçek değerler
    fig.add_trace(go.Scatter(
        x=last_n_predictions.index,
        y=last_n_predictions['target'],
        mode='lines+markers',
        name='Gerçekleşen',
        line=dict(width=2, color='blue')
    ))
    
    fig.update_layout(
        title=f"Son {n_days} Günün BIST 100 Yön Değişimleri",
        xaxis_title="Tarih",
        yaxis_title="Yön (1: Artış, 0: Azalış)",
        legend_title="Veri",
        height=400
    )
    
    return fig

def prepare_display_data(raw_data, view_period, normalize=False):
    """
    Görselleştirme için veriyi hazırlar (zaman aralığı filtreler ve isteğe bağlı normalize eder)
    
    Args:
        raw_data: Ham veri DataFrame'i
        view_period: Görüntüleme periyodu (string)
        normalize: Verilerin normalize edilip edilmeyeceği (boolean)
        
    Returns:
        Plotly figure
    """
    # Zaman aralığını belirle
    if view_period == "Son 1 Ay":
        data_view = raw_data.tail(30)
    elif view_period == "Son 3 Ay":
        data_view = raw_data.tail(90)
    elif view_period == "Son 6 Ay":
        data_view = raw_data.tail(180)
    elif view_period == "Son 1 Yıl":
        data_view = raw_data.tail(365)
    else:
        data_view = raw_data
    
    # Grafikler için veriyi hazırla
    if normalize:
        # Veriyi normalize et
        plot_data = data_view.copy()
        for col in plot_data.columns:
            plot_data[col] = plot_data[col] / plot_data[col].iloc[0] * 100
    else:
        plot_data = data_view
    
    return plot_data, data_view

def plot_lag_correlation_heatmap(lag_corr_pivot, title="Gecikmeli Etki Korelasyon Isı Haritası"):
    """
    Farklı değişkenler ve gecikmeli günler için korelasyon ısı haritası oluşturur.
    
    Args:
        lag_corr_pivot: Değişken ve lag bazında pivot tablo şeklinde korelasyon verileri
        title: Grafik başlığı
        
    Returns:
        Plotly figure
    """
    fig = px.imshow(
        lag_corr_pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin='lower',
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Gecikme (Gün)",
        yaxis_title="Değişken",
        height=500
    )
    
    return fig

def plot_lag_effect_line(lag_corr_df, variables=None, title="Değişkenlerin Gecikmeli Etkileri"):
    """
    Değişkenlerin farklı gecikme günlerindeki korelasyonlarını çizgi grafiği olarak gösterir.
    
    Args:
        lag_corr_df: Değişken, gecikme ve korelasyon sütunlarını içeren DataFrame
        variables: Görselleştirilecek değişkenler listesi (None ise tümü gösterilir)
        title: Grafik başlığı
        
    Returns:
        Plotly figure
    """
    if variables is None:
        variables = lag_corr_df['variable'].unique()
    
    filtered_df = lag_corr_df[lag_corr_df['variable'].isin(variables)]
    
    fig = px.line(
        filtered_df,
        x="lag_days",
        y="correlation",
        color="variable",
        title=title,
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Gecikme (Gün)",
        yaxis_title="BIST100 ile Korelasyon",
        legend_title="Değişken",
        height=500
    )
    
    # Referans çizgisi ekle (0 korelasyon)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig

def plot_rolling_correlation(df, target_col="BIST100", window=90, variables=None):
    """
    Seçilen değişkenlerin hedef değişken ile hareketli korelasyonunu gösterir.
    
    Args:
        df: Ham fiyat verilerini içeren DataFrame
        target_col: Hedef değişken sütunu
        window: Hareketli korelasyon pencere büyüklüğü (gün sayısı)
        variables: Görselleştirilecek değişkenler listesi (None ise tümü gösterilir)
        
    Returns:
        Plotly figure
    """
    # Günlük değişim oranlarını hesapla
    df_pct = df.pct_change().dropna()
    
    # Görselleştirilecek değişkenleri belirle
    if variables is None:
        variables = [col for col in df_pct.columns if col != target_col]
    
    # Hareketli korelasyonları hesapla
    rolling_corrs = pd.DataFrame(index=df_pct.index[window-1:])
    
    for col in variables:
        rolling_corrs[col] = df_pct[target_col].rolling(window=window).corr(df_pct[col])
    
    # Çizgi grafiğini oluştur
    fig = go.Figure()
    
    for col in rolling_corrs.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_corrs.index,
                y=rolling_corrs[col],
                mode='lines',
                name=col
            )
        )
    
    fig.update_layout(
        title=f"{window} Günlük Hareketli Korelasyon ({target_col} ile)",
        xaxis_title="Tarih",
        yaxis_title="Korelasyon",
        legend_title="Değişken",
        height=500
    )
    
    # Referans çizgisi ekle (0 korelasyon)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig

def plot_global_variables_dashboard(df, period="1Y"):
    """
    Global değişkenlerin tümünü gösteren dashboard tarzı bir görsel oluşturur.
    
    Args:
        df: Ham fiyat verilerini içeren DataFrame
        period: Gösterilecek dönem ('1M', '3M', '6M', '1Y', 'All')
        
    Returns:
        Plotly figure
    """
    # Dönemi belirle
    if period == "1M":
        data_view = df.tail(30)
    elif period == "3M":
        data_view = df.tail(90)
    elif period == "6M":
        data_view = df.tail(180)
    elif period == "1Y":
        data_view = df.tail(365)
    else:
        data_view = df
    
    # Normalize et (başlangıç = 100)
    norm_data = data_view.copy()
    for col in norm_data.columns:
        norm_data[col] = norm_data[col] / norm_data[col].iloc[0] * 100
    
    # Günlük değişim
    pct_data = data_view.pct_change().dropna() * 100
    
    # Dashboard oluştur (3x2 grid)
    fig = make_subplots(
        rows=3, 
        cols=2,
        subplot_titles=(
            "Normalize Fiyatlar (Başlangıç=100)", 
            "Günlük Değişimler (%)",
            "Korelasyon Matrisi",
            "BIST100 ile Korelasyon",
            "Volatilite (Std, 30 gün)",
            "US10Y, VIX ve BIST100 İlişkisi"
        )
    )
    
    # Renk skalası
    colors = {
        "BIST100": "#1f77b4",  # Mavi
        "Gold": "#ff7f0e",     # Turuncu
        "Oil": "#2ca02c",      # Yeşil
        "USDTRY": "#d62728",   # Kırmızı
        "US10Y": "#9467bd",    # Mor
        "NatGas": "#8c564b",   # Kahverengi
        "VIX": "#e377c2"       # Pembe
    }
    
    # 1. Normalize fiyatlar
    for col in norm_data.columns:
        color = colors.get(col, None)
        fig.add_trace(
            go.Scatter(
                x=norm_data.index, 
                y=norm_data[col], 
                name=col,
                mode='lines',
                line=dict(color=color)
            ),
            row=1, col=1
        )
    
    # 2. Günlük değişimler
    for col in pct_data.columns:
        color = colors.get(col, None)
        fig.add_trace(
            go.Scatter(
                x=pct_data.index, 
                y=pct_data[col], 
                name=col,
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Korelasyon matrisi
    corr_matrix = data_view.corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            showscale=False
        ),
        row=2, col=1
    )
    
    # 4. BIST100 ile korelasyon
    bist_corr = corr_matrix['BIST100'].drop('BIST100').sort_values()
    
    colors_bar = []
    for col in bist_corr.index:
        colors_bar.append(colors.get(col, '#636EFA'))
    
    fig.add_trace(
        go.Bar(
            y=bist_corr.index,
            x=bist_corr.values,
            orientation='h',
            marker_color=colors_bar,
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 5. Volatilite
    vol_data = pd.DataFrame()
    for col in data_view.columns:
        vol_data[col] = data_view[col].pct_change().rolling(30).std() * 100
    
    vol_data = vol_data.dropna()
    
    for col in vol_data.columns:
        color = colors.get(col, None)
        fig.add_trace(
            go.Scatter(
                x=vol_data.index, 
                y=vol_data[col], 
                name=col,
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 6. US10Y, VIX ve BIST100 İlişkisi - Scatter Plot
    if 'US10Y' in data_view.columns and 'VIX' in data_view.columns:
        fig.add_trace(
            go.Scatter(
                x=data_view['US10Y'],
                y=data_view['BIST100'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data_view['VIX'],
                    colorscale='Viridis',
                    colorbar=dict(
                        title="VIX",
                        len=0.5,  # Reduce length to 50% of the plot
                        thickness=15,  # Reduce thickness
                        y=0.5,  # Center vertically
                        yanchor="middle",
                        x=1.05,  # Position slightly to the right
                        xanchor="left"
                    ),
                    showscale=True
                ),
                text=data_view.index.strftime('%Y-%m-%d'),
                name='US10Y vs BIST100',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Trend çizgisi
        x = data_view['US10Y'].values
        y = data_view['BIST100'].values
        
        # Eksik değerleri kaldır
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) > 1:
            # Trend çizgisi için linear regresyon
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Trend çizgisini çiz
            x_range = np.linspace(min(x), max(x), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.5)', width=2, dash='dash'),
                    name='Trend',
                    showlegend=False
                ),
                row=3, col=2
            )
    
    # Layout ayarları
    fig.update_layout(
        title=f"Global Değişkenler Dashboard - {period}",
        height=900,
        width=1200
    )
    
    # X ekseni başlığı
    for i in range(1, 4):
        for j in range(1, 3):
            if i == 3 and j == 2:
                fig.update_xaxes(title_text="US10Y (%)", row=i, col=j)
            else:
                fig.update_xaxes(title_text="Tarih", row=i, col=j)
    
    # Y ekseni başlıkları
    fig.update_yaxes(title_text="Endeks Değeri", row=1, col=1)
    fig.update_yaxes(title_text="Günlük Değişim (%)", row=1, col=2)
    fig.update_yaxes(title_text="Değişken", row=2, col=1)
    fig.update_yaxes(title_text="Değişken", row=2, col=2)
    fig.update_yaxes(title_text="Volatilite (%)", row=3, col=1)
    fig.update_yaxes(title_text="BIST 100", row=3, col=2)
    
    # X ekseni başlığı (özel durumlar)
    fig.update_xaxes(title_text="Korelasyon", row=2, col=2)
    
    return fig

def plot_enhanced_correlation_matrix(data, title="BIST100 ve Emtialar/Göstergeler Arasındaki Korelasyon"):
    """
    Geliştirilmiş, daha görsel ve etkileyici korelasyon matrisi oluşturur.
    
    Args:
        data: Pandas DataFrame - içerisinde korelasyon hesaplanacak veriler
        title: Grafik başlığı
        
    Returns:
        Plotly figure
    """
    # Korelasyon matrisini hesapla
    corr_matrix = data.corr()
    
    # BIST100 ile olan korelasyonları al
    bist_correlations = corr_matrix['BIST100'].drop('BIST100')
    
    # Korelasyonları mutlak değere göre sırala
    sorted_correlations = bist_correlations.abs().sort_values(ascending=False)
    sorted_bist_correlations = bist_correlations[sorted_correlations.index]
    
    # Pozitif ve negatif ayırma
    positive_corrs = sorted_bist_correlations[sorted_bist_correlations > 0]
    negative_corrs = sorted_bist_correlations[sorted_bist_correlations < 0]
    
    # Renk skalası
    colors = []
    for corr in sorted_bist_correlations:
        if corr > 0.5:  # Güçlü pozitif
            colors.append('rgba(0, 128, 0, 0.8)')  # Koyu yeşil
        elif corr > 0:  # Zayıf pozitif
            colors.append('rgba(144, 238, 144, 0.8)')  # Açık yeşil
        elif corr > -0.5:  # Zayıf negatif
            colors.append('rgba(255, 165, 0, 0.8)')  # Turuncu
        else:  # Güçlü negatif
            colors.append('rgba(255, 0, 0, 0.8)')  # Kırmızı
    
    # Çubuk grafik
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=sorted_bist_correlations.index,
            y=sorted_bist_correlations.values,
            marker_color=colors,
            text=[f"{corr:.2f}" for corr in sorted_bist_correlations.values],
            textposition='auto'
        )
    )
    
    # Açıklama metni
    explanation = """
    <b>Korelasyon Yorumu:</b><br>
    <span style='color:green'>Yeşil (Pozitif Korelasyon)</span>: Bu emtia/gösterge arttığında, BIST100 de artma eğilimindedir.<br>
    <span style='color:red'>Kırmızı (Negatif Korelasyon)</span>: Bu emtia/gösterge arttığında, BIST100 düşme eğilimindedir.<br>
    <br>
    <b>Korelasyon Gücü:</b><br>
    0.7-1.0: Güçlü ilişki<br>
    0.4-0.7: Orta düzeyde ilişki<br>
    0.0-0.4: Zayıf ilişki
    """
    
    # Eşik çizgileri
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="green"),
        x0=-0.5, x1=len(sorted_bist_correlations) - 0.5, y0=0.7, y1=0.7
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="lightgreen"),
        x0=-0.5, x1=len(sorted_bist_correlations) - 0.5, y0=0.4, y1=0.4
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="orange"),
        x0=-0.5, x1=len(sorted_bist_correlations) - 0.5, y0=0, y1=0
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="lightcoral"),
        x0=-0.5, x1=len(sorted_bist_correlations) - 0.5, y0=-0.4, y1=-0.4
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="red"),
        x0=-0.5, x1=len(sorted_bist_correlations) - 0.5, y0=-0.7, y1=-0.7
    )
    
    # Düzenleme
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Emtialar ve Göstergeler",
        yaxis_title="BIST100 ile Korelasyon",
        yaxis=dict(
            range=[-1, 1],
            tickvals=[-1, -0.7, -0.4, 0, 0.4, 0.7, 1],
            ticktext=["-1.0<br>(Tam Ters)", "-0.7<br>(Güçlü Ters)", "-0.4<br>(Orta Ters)", 
                      "0<br>(İlişkisiz)", "0.4<br>(Orta Aynı)", "0.7<br>(Güçlü Aynı)", "1.0<br>(Tam Aynı)"]
        ),
        height=600,
        margin=dict(l=50, r=50, t=90, b=50),
        annotations=[
            dict(
                x=1.01,
                y=0,
                xref="paper",
                yref="paper",
                text=explanation,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=10,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

    """
    BIST100 tahminlerinin animasyonlu zaman serisi grafiğini oluşturur.
    
    Args:
        y_true: Gerçek değerler (1: artış, 0: azalış)
        y_pred: Tahmin değerleri (1: artış, 0: azalış)
        y_proba: Artış olasılığı değerleri (0-1 arası)
        dates: Tarih dizisi
        window: Gösterilecek gün sayısı
        
    Returns:
        Plotly figure
    """
    # Verileri DataFrame'e dönüştür
    df = pd.DataFrame({
        'date': dates,
        'actual': y_true,
        'prediction': y_pred,
        'probability': y_proba,
        'correct': (y_true == y_pred).astype(int)
    })
    
    # BIST100 değerleri için yapay veri oluştur (gerçek değeri bilmiyoruz)
    df['bist100_value'] = 1000  # başlangıç değeri
    
    # Artış/azalış durumuna göre BIST100 değerlerini hesapla
    # Artışta %1 yükseliş, azalışta %1 düşüş varsayalım
    for i in range(1, len(df)):
        if df.iloc[i-1]['actual'] == 1:  # Artış
            df.loc[df.index[i], 'bist100_value'] = df.iloc[i-1]['bist100_value'] * 1.01
        else:  # Azalış
            df.loc[df.index[i], 'bist100_value'] = df.iloc[i-1]['bist100_value'] * 0.99
    
    # Animasyon için frame'ler oluştur
    frames = []
    for i in range(window, len(df)):
        subset = df.iloc[i-window:i+1]
        
        # Doğru tahmin sayısını hesapla
        correct_count = subset['correct'].sum()
        accuracy = correct_count / len(subset)
        
        # Ana çizgi grafiği
        data = [
            # BIST100 değerleri
            go.Scatter(
                x=subset['date'],
                y=subset['bist100_value'],
                mode='lines',
                name='BIST100',
                line=dict(color='blue', width=2)
            ),
            # Tahmin değerleri
            go.Scatter(
                x=[subset['date'].iloc[-1]],
                y=[subset['bist100_value'].iloc[-1]],
                mode='markers',
                name='Artış Tahmini' if subset['prediction'].iloc[-1] == 1 else 'Düşüş Tahmini',
                marker=dict(
                    color='green' if subset['prediction'].iloc[-1] == 1 else 'red',
                    size=15,
                    symbol='triangle-up' if subset['prediction'].iloc[-1] == 1 else 'triangle-down',
                    line=dict(color='black', width=1)
                )
            ),
            # Olasılık değeri
            go.Scatter(
                x=subset['date'],
                y=subset['probability'] * subset['bist100_value'].max() * 0.2 + subset['bist100_value'].min() * 0.8,
                mode='lines',
                name='Artış Olasılığı',
                line=dict(color='purple', dash='dash'),
                yaxis='y2'
            )
        ]
        
        # Doğruluk bilgisini metin olarak ekle
        annotations = [
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'<b>Son {window} Gün Doğruluk: {accuracy:.1%}</b>',
                showarrow=False,
                font=dict(size=14, color='black'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=5,
                align='left'
            ),
            dict(
                x=0.05,
                y=0.85,
                xref='paper',
                yref='paper',
                text=f'<b>Tarih: {subset["date"].iloc[-1].strftime("%d.%m.%Y")}</b>',
                showarrow=False,
                font=dict(size=14, color='black'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=5,
                align='left'
            ),
            dict(
                x=0.05,
                y=0.75,
                xref='paper',
                yref='paper',
                text=f'<b>Yarınki Tahmin: {"Artış 📈" if subset["prediction"].iloc[-1] == 1 else "Düşüş 📉"}</b>',
                showarrow=False,
                font=dict(
                    size=14, 
                    color='green' if subset["prediction"].iloc[-1] == 1 else 'red'
                ),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=5,
                align='left'
            ),
            dict(
                x=0.05,
                y=0.65,
                xref='paper',
                yref='paper',
                text=f'<b>Olasılık: {subset["probability"].iloc[-1]:.1%}</b>',
                showarrow=False,
                font=dict(size=14, color='purple'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=5,
                align='left'
            )
        ]
        
        frames.append(
            go.Frame(
                data=data,
                layout=go.Layout(
                    annotations=annotations
                ),
                name=str(i)
            )
        )
    
    # Ana grafik
    fig = go.Figure(
        data=[
            # BIST100 değerleri
            go.Scatter(
                x=df['date'].iloc[:window],
                y=df['bist100_value'].iloc[:window],
                mode='lines',
                name='BIST100',
                line=dict(color='blue', width=2)
            ),
            # Tahmin değerleri
            go.Scatter(
                x=[df['date'].iloc[window-1]],
                y=[df['bist100_value'].iloc[window-1]],
                mode='markers',
                name='Artış Tahmini' if df['prediction'].iloc[window-1] == 1 else 'Düşüş Tahmini',
                marker=dict(
                    color='green' if df['prediction'].iloc[window-1] == 1 else 'red',
                    size=15,
                    symbol='triangle-up' if df['prediction'].iloc[window-1] == 1 else 'triangle-down',
                    line=dict(color='black', width=1)
                )
            ),
            # Olasılık değeri
            go.Scatter(
                x=df['date'].iloc[:window],
                y=df['probability'].iloc[:window] * df['bist100_value'].iloc[:window].max() * 0.2 + df['bist100_value'].iloc[:window].min() * 0.8,
                mode='lines',
                name='Artış Olasılığı',
                line=dict(color='purple', dash='dash'),
                yaxis='y2'
            )
        ],
        layout=go.Layout(
            title={
                'text': 'BIST100 Tahmin Animasyonu',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            xaxis=dict(
                title='Tarih',
                range=[df['date'].iloc[0], df['date'].iloc[window-1]]
            ),
            yaxis=dict(
                title='BIST100 Değeri'
            ),
            yaxis2=dict(
                title='Artış Olasılığı',
                title_font=dict(color='purple'),
                tickfont=dict(color='purple'),
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='▶️ Oynat',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='⏸️ Durdur',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=True),
                                mode='immediate',
                                transition=dict(duration=0)
                            )]
                        )
                    ],
                    direction='left',
                    pad=dict(l=10, r=10, t=10, b=10),
                    x=0.1,
                    y=0,
                    xanchor='right',
                    yanchor='top'
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method='animate',
                            args=[
                                [str(i)],
                                dict(
                                    mode='immediate',
                                    frame=dict(duration=200, redraw=True),
                                    transition=dict(duration=0)
                                )
                            ],
                            label=str(df['date'].iloc[i].strftime("%d.%m.%Y"))
                        )
                        for i in range(window, len(df))
                    ],
                    active=0,
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix='Tarih: ',
                        visible=True,
                        xanchor='center'
                    ),
                    pad=dict(t=30),
                    len=0.9,
                    x=0.1,
                    y=0,
                    xanchor='left',
                    yanchor='top'
                )
            ],
            height=700,
            margin=dict(l=50, r=50, t=90, b=100),
            annotations=[
                dict(
                    x=0.05,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Son {window} Gün Doğruluk: {df["correct"].iloc[:window].mean():.1%}</b>',
                    showarrow=False,
                    font=dict(size=14, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=5,
                    align='left'
                ),
                dict(
                    x=0.05,
                    y=0.85,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Tarih: {df["date"].iloc[window-1].strftime("%d.%m.%Y")}</b>',
                    showarrow=False,
                    font=dict(size=14, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=5,
                    align='left'
                ),
                dict(
                    x=0.05,
                    y=0.75,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Yarınki Tahmin: {"Artış 📈" if df["prediction"].iloc[window-1] == 1 else "Düşüş 📉"}</b>',
                    showarrow=False,
                    font=dict(
                        size=14, 
                        color='green' if df["prediction"].iloc[window-1] == 1 else 'red'
                    ),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=5,
                    align='left'
                ),
                dict(
                    x=0.05,
                    y=0.65,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Olasılık: {df["probability"].iloc[window-1]:.1%}</b>',
                    showarrow=False,
                    font=dict(size=14, color='purple'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=5,
                    align='left'
                )
            ]
        ),
        frames=frames
    )
    
    return fig