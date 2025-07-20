import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from scipy import stats
import io
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tema hitam-biru
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #1e40af, #3b82f6);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .chart-navbar {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .simulation-results {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .export-container {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DataProcessor:
    """Class untuk memproses data dan mengurangi duplikasi code"""
    
    @staticmethod
    def load_sample_data():
        """Membuat data sampel untuk demonstrasi"""
        np.random.seed(42)
        
        years = [2021, 2022, 2023, 2024]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        products = ['Salsa, 418ml', 'Tomato Sauce, 340g', 'Pasta, 500g', 'Rice, 1kg', 'Oil, 250ml']
        
        data = []
        for year in years:
            for month in months:
                for product in products:
                    if product == 'Salsa, 418ml':
                        base_price = 7.5
                        price = np.random.normal(base_price, 0.8)
                        price = max(5.0, min(10.0, price))
                    else:
                        price = np.random.uniform(3.0, 15.0)
                    
                    tax_rate = np.random.uniform(0.08, 0.12)
                    value_after_tax = price * (1 + tax_rate)
                    
                    data.append({
                        'Year': year,
                        'Month': month,
                        'GEO': f'Provinsi {np.random.randint(1, 6)}',
                        'Product Category': 'Canned Goods' if 'Salsa' in product else 'Food Items',
                        'Products': product,
                        'VALUE': round(price, 2),
                        'Taxable': 'Yes',
                        'Total Tax Rate': round(tax_rate * 100, 1),
                        'Value After Tax': round(value_after_tax, 2),
                        'Essential': np.random.choice(['Essential', 'Non-Essential']),
                        'Coordinate': f"{np.random.randint(-10, 10)},{np.random.randint(100, 120)}",
                        'UOM': 'Dollars'
                    })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def filter_data(df, selected_product, year_range):
        """Filter data berdasarkan parameter user"""
        return df[
            (df['Products'] == selected_product) & 
            (df['Year'] >= year_range[0]) & 
            (df['Year'] <= year_range[1])
        ]
    
    @staticmethod
    def calculate_statistics(data):
        """Hitung statistik dasar untuk data"""
        return {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min()
        }

class MonteCarloEngine:
    """Class untuk simulasi Monte Carlo"""
    
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        random.seed(42)
        np.random.seed(42)
    
    def create_frequency_distribution(self, data):
        """Membuat tabel distribusi frekuensi"""
        min_val = data.min()
        max_val = data.max()
        class_width = (max_val - min_val) / self.num_classes
        
        classes, frequencies, midpoints, probabilities = [], [], [], []
        
        for i in range(self.num_classes):
            lower = min_val + i * class_width
            upper = min_val + (i + 1) * class_width
            midpoint = (lower + upper) / 2
            
            if i == self.num_classes - 1:
                count = len(data[(data >= lower) & (data <= upper)])
            else:
                count = len(data[(data >= lower) & (data < upper)])
            
            classes.append(f"{lower:.2f}% - {upper:.2f}%")
            frequencies.append(count)
            midpoints.append(midpoint)
            probabilities.append(round(count/len(data), 4))
        
        # Probabilitas kumulatif
        cumulative_probs = np.cumsum(probabilities).round(4).tolist()
        
        df_freq = pd.DataFrame({
            'X(%)': classes,
            'Frekuensi': frequencies,
            'Titik Tengah': [round(x, 2) for x in midpoints],
            'Probabilitas': probabilities,
            'Probabilitas Kumulatif': cumulative_probs
        })
        
        return df_freq, midpoints, probabilities
    
    def run_simulation(self, midpoints, probabilities, num_simulations):
        """Jalankan simulasi Monte Carlo"""
        random_numbers = [random.random() for _ in range(num_simulations)]
        random_percentages = [r * 100 for r in random_numbers]
        
        # Tabel persiapan
        prep_table = pd.DataFrame({
            'Zi': range(1, num_simulations + 1),
            'Ui': [round(r, 4) for r in random_numbers],
            'Ui * 100': [round(r, 2) for r in random_percentages]
        })
        
        # Mapping Monte Carlo
        cumulative_probs_pct = np.cumsum(probabilities).tolist()
        cumulative_probs_pct = [prob * 100 for prob in cumulative_probs_pct]
        
        simulated_prices = []
        for rand_pct in random_percentages:
            for i, cum_prob in enumerate(cumulative_probs_pct):
                if rand_pct <= cum_prob:
                    simulated_prices.append(midpoints[i])
                    break
        
        # Format hasil simulasi
        simulation_results = []
        prev_price = simulated_prices[0] if simulated_prices else 7.0
        
        for i, price in enumerate(simulated_prices):
            fluctuation = 0.0 if i == 0 else round(price - prev_price, 2)
            tax_rate = 0.12
            price_before_tax = round(price / (1 + tax_rate), 2)
            
            simulation_results.append({
                'Harga Simulasi': round(price, 2),
                'Fluktuasi Harga': fluctuation,
                'Harga Sebelum Pajak': price_before_tax
            })
            
            prev_price = price
        
        return pd.DataFrame(simulation_results), prep_table

class ChartGenerator:
    """Class untuk membuat visualisasi"""
    
    @staticmethod
    def create_scatter_plot(df_simulation, df_prep):
        """Scatter Plot Titik Acak Monte Carlo"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df_prep['Ui'],
                y=df_simulation['Harga Simulasi'],
                mode='markers',
                name='Titik Acak MC',
                marker=dict(
                    color=df_simulation['Harga Simulasi'],
                    colorscale='Viridis',
                    size=8,
                    opacity=0.7,
                    colorbar=dict(title="Harga ($)")
                ),
                text=[f"Simulasi {i+1}<br>Random: {u:.3f}<br>Harga: ${p:.2f}" 
                      for i, (u, p) in enumerate(zip(df_prep['Ui'], df_simulation['Harga Simulasi']))],
                hovertemplate='%{text}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title="🎲 Scatter Plot Titik Acak Monte Carlo",
            xaxis_title="Bilangan Acak Ui",
            yaxis_title="Harga Simulasi ($)",
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 0.8)',
            font_color='white'
        )
        
        return fig
    
    @staticmethod
    def create_convergence_plot(df_simulation, historical_mean):
        """Plot Konvergensi Mean"""
        cumulative_mean = df_simulation['Harga Simulasi'].expanding().mean()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_mean)+1)),
                y=cumulative_mean,
                mode='lines',
                name='Konvergensi MC',
                line=dict(color='#3b82f6', width=2)
            )
        )
        
        fig.add_hline(
            y=historical_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean Historis: ${historical_mean:.2f}"
        )
        
        fig.update_layout(
            title="📈 Konvergensi Mean (Stabilitas Simulasi)",
            xaxis_title="Iterasi Simulasi",
            yaxis_title="Mean Kumulatif ($)",
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 0.8)',
            font_color='white'
        )
        
        return fig
    
    @staticmethod
    def create_histogram(df_simulation, df_historical):
        """Histogram Distribusi Hasil"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=df_simulation['Harga Simulasi'],
                nbinsx=10,
                name='Distribusi MC',
                marker_color='#60a5fa',
                opacity=0.7,
                histnorm='probability'
            )
        )
        
        fig.add_trace(
            go.Histogram(
                x=df_historical['VALUE'],
                nbinsx=10,
                name='Distribusi Historis',
                marker_color='#10b981',
                opacity=0.5,
                histnorm='probability'
            )
        )
        
        fig.update_layout(
            title="📊 Histogram Distribusi Hasil",
            xaxis_title="Harga ($)",
            yaxis_title="Probabilitas",
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 0.8)',
            font_color='white',
            barmode='overlay'
        )
        
        return fig
    
    @staticmethod
    def create_timeseries(df_simulation, df_historical):
        """Time Series Perbandingan"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df_simulation)+1)),
                y=df_simulation['Harga Simulasi'],
                mode='lines+markers',
                name='Simulasi MC',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=4)
            )
        )
        
        # Sample historical data untuk perbandingan
        historical_sample = df_historical['VALUE'].sample(len(df_simulation), replace=True).reset_index(drop=True)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(historical_sample)+1)),
                y=historical_sample,
                mode='lines',
                name='Trend Historis',
                line=dict(color='#10b981', width=2, dash='dot')
            )
        )
        
        fig.update_layout(
            title="⏱️ Time Series Simulasi vs Historis",
            xaxis_title="Waktu",
            yaxis_title="Harga ($)",
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 0.8)',
            font_color='white'
        )
        
        return fig

class ExportManager:
    """Class untuk export data ke Excel"""
    
    @staticmethod
    def create_excel_export(df_freq, df_prep, df_simulation, statistics):
        """Buat file Excel dengan multiple sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Distribusi Frekuensi
            df_freq.to_excel(writer, sheet_name='Distribusi Frekuensi', index=False)
            
            # Sheet 2: Persiapan Monte Carlo
            df_prep.to_excel(writer, sheet_name='Persiapan Monte Carlo', index=False)
            
            # Sheet 3: Hasil Simulasi
            df_simulation.to_excel(writer, sheet_name='Hasil Simulasi', index=False)
            
            # Sheet 4: Statistik
            stats_df = pd.DataFrame({
                'Metrik': ['Jumlah Data', 'Rata-rata', 'Standar Deviasi', 'Minimum', 'Maksimum', 'Range'],
                'Nilai': [statistics['count'], statistics['mean'], statistics['std'], 
                         statistics['min'], statistics['max'], statistics['range']]
            })
            stats_df.to_excel(writer, sheet_name='Statistik', index=False)
        
        return output.getvalue()

def main():
    # Header
    st.markdown('<div class="main-header"><h1>🎲 Simulasi Monte Carlo - Analisis Harga Produk</h1></div>', 
                unsafe_allow_html=True)
    
    # Initialize processors
    data_processor = DataProcessor()
    monte_carlo = MonteCarloEngine()
    chart_gen = ChartGenerator()
    export_manager = ExportManager()
    
    # Sidebar untuk input parameter
    with st.sidebar:
        st.markdown("### 📊 Parameter Simulasi")
        
        # Upload file atau gunakan data sampel
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
        
        if uploaded_file is None:
            st.info("Menggunakan data sampel untuk demonstrasi")
            df = data_processor.load_sample_data()
        else:
            df = pd.read_excel(uploaded_file)
        
        # Filter produk
        products = df['Products'].unique()
        selected_product = st.selectbox("Pilih Produk:", products)
        
        # Filter tahun
        years = sorted(df['Year'].unique())
        year_range = st.slider("Rentang Tahun:", 
                              min_value=min(years), 
                              max_value=max(years), 
                              value=(min(years), max(years)))
        
        # Jumlah simulasi
        num_simulations = st.number_input("Jumlah Simulasi:", 
                                        min_value=10, max_value=1000, 
                                        value=30, step=10)
        
        # Tombol simulasi
        simulate_button = st.button("🚀 Mulai Simulasi", 
                                  use_container_width=True,
                                  type="primary")
    
    # Filter data
    filtered_data = data_processor.filter_data(df, selected_product, year_range)
    
    # Tampilan awal - penjelasan aplikasi
    if not simulate_button:
        st.markdown("### 📖 Tentang Aplikasi")
        st.markdown("""
        **Aplikasi Simulasi Monte Carlo** ini dirancang untuk menganalisis dan memprediksi harga produk 
        berdasarkan data historis menggunakan metode Monte Carlo. Simulasi ini membantu dalam:
        
        - 📊 **Analisis Distribusi Harga**: Memahami pola sebaran harga historis
        - 🎯 **Prediksi Harga**: Simulasi kemungkinan harga di masa depan
        - 📈 **Analisis Fluktuasi**: Menghitung volatilitas dan perubahan harga
        - 💰 **Perhitungan Pajak**: Konversi harga sebelum dan setelah pajak
        """)
        
        st.markdown("### 📋 Format File Excel yang Diperlukan")
        st.markdown("""
        File Excel Anda harus memiliki kolom berikut:
        
        | **Kolom** | **Deskripsi** | **Contoh** |
        |-----------|---------------|------------|
        | `Year` | Tahun data | 2021, 2022, 2023 |
        | `Month` | Bulan | Jan, Feb, Mar |
        | `Products` | Nama produk | Salsa, 418ml |
        | `VALUE` | Harga produk | 7.50, 8.20 |
        | `Total Tax Rate` | Tarif pajak (%) | 11.0, 12.5 |
        """)
        
        st.markdown("### 🚀 Cara Menggunakan")
        st.markdown("""
        1. **Upload File Excel** atau gunakan data sampel yang tersedia
        2. **Pilih Parameter** di sidebar (produk, tahun, jumlah simulasi)
        3. **Klik "Mulai Simulasi"** untuk menjalankan Monte Carlo
        4. **Analisis Hasil** melalui tabel dan visualisasi yang dihasilkan
        5. **Export Hasil** ke Excel untuk analisis lebih lanjut
        """)
        
        st.info("💡 **Tips**: Semakin banyak data historis dan simulasi, hasil prediksi akan semakin akurat!")
    
    else:
        # Preview data
        st.markdown("### 📋 Data yang Digunakan")
        st.write(f"**Produk**: {selected_product} | **Periode**: {year_range[0]} - {year_range[1]} | **Total Data**: {len(filtered_data)} record")
        
        with st.expander("Lihat Preview Data"):
            st.dataframe(filtered_data.head(10), use_container_width=True)
    
    if simulate_button and len(filtered_data) > 0:
        with st.spinner('Menjalankan simulasi Monte Carlo...'):
            # Ekstrak data harga
            price_data = filtered_data['VALUE']
            
            # Hitung statistik
            statistics = data_processor.calculate_statistics(price_data)
            
            # Buat distribusi frekuensi
            df_freq, midpoints, probabilities = monte_carlo.create_frequency_distribution(price_data)
            
            # Jalankan simulasi
            df_simulation, df_prep = monte_carlo.run_simulation(midpoints, probabilities, num_simulations)
            
            # Tampilkan hasil
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📊 Tabel Distribusi Frekuensi")
                
                # Statistik
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Banyak Data", f"{statistics['count']}")
                    st.metric("Rentang", f"{statistics['range']:.2f}")
                with stat_col2:
                    st.metric("Data Max", f"{statistics['max']:.2f}")
                    st.metric("Banyak Kelas", f"{len(df_freq)}")
                with stat_col3:
                    st.metric("Data Min", f"{statistics['min']:.2f}")
                    st.metric("Rata-rata", f"{statistics['mean']:.2f}")
                
                st.dataframe(df_freq[['X(%)', 'Frekuensi']], use_container_width=True, hide_index=True)
                
                with st.expander("📈 Detail Distribusi & Probabilitas"):
                    st.dataframe(df_freq, use_container_width=True)
                
                st.markdown("### 🎲 Tabel Persiapan Monte Carlo")
                st.dataframe(df_prep.head(10), use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Statistik Simulasi")
                sim_stats = data_processor.calculate_statistics(df_simulation['Harga Simulasi'])
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Rata-rata Harga", f"${sim_stats['mean']:.2f}")
                    st.metric("Harga Minimum", f"${sim_stats['min']:.2f}")
                
                with metrics_col2:
                    st.metric("Standar Deviasi", f"${sim_stats['std']:.2f}")
                    st.metric("Harga Maksimum", f"${sim_stats['max']:.2f}")
            
            # Tabel hasil simulasi
            st.markdown('<div class="simulation-results">', unsafe_allow_html=True)
            st.markdown("### 🎯 Hasil Simulasi Monte Carlo")
            st.dataframe(df_simulation, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chart Navigation
            st.markdown('<div class="chart-navbar">', unsafe_allow_html=True)
            st.markdown("### 📊 Visualisasi Monte Carlo")
            
            # Navbar untuk pilih chart
            chart_options = [
                "🎲 Scatter Plot Titik Acak", 
                "📈 Konvergensi Mean", 
                "📊 Histogram Distribusi", 
                "⏱️ Time Series"
            ]
            
            selected_chart = st.radio(
                "Pilih Visualisasi:",
                chart_options,
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Tampilkan chart yang dipilih
            if selected_chart == "🎲 Scatter Plot Titik Acak":
                fig = chart_gen.create_scatter_plot(df_simulation, df_prep)
            elif selected_chart == "📈 Konvergensi Mean":
                fig = chart_gen.create_convergence_plot(df_simulation, statistics['mean'])
            elif selected_chart == "📊 Histogram Distribusi":
                fig = chart_gen.create_histogram(df_simulation, filtered_data)
            else:  # Time Series
                fig = chart_gen.create_timeseries(df_simulation, filtered_data)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export Section
            st.markdown('<div class="export-container">', unsafe_allow_html=True)
            st.markdown("### 📥 Export Hasil")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                excel_data = export_manager.create_excel_export(
                    df_freq, df_prep, df_simulation, sim_stats
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"monte_carlo_results_{timestamp}.xlsx"
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Insights
            st.markdown("### 💡 Insight Simulasi")
            avg_fluctuation = df_simulation['Fluktuasi Harga'].mean()
            
            st.write(f"""
            **Hasil Analisis:**
            - Harga rata-rata simulasi: ${sim_stats['mean']:.2f}
            - Harga rata-rata historis: ${statistics['mean']:.2f}
            - Selisih: ${abs(sim_stats['mean'] - statistics['mean']):.2f}
            - Fluktuasi rata-rata: ${avg_fluctuation:.2f}
            - Volatilitas (CV): {(sim_stats['std']/sim_stats['mean'])*100:.1f}%
            """)

if __name__ == "__main__":
    main()