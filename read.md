# Simulator Harga Monte Carlo

Aplikasi ini merupakan alat analisis harga produk menggunakan metode Monte Carlo untuk memprediksi fluktuasi harga berdasarkan data historis.

## Fitur Utama

- **Analisis Distribusi Harga**: Memvisualisasikan pola harga historis
- **Prediksi Harga**: Simulasi skenario harga masa depan
- **Analisis Volatilitas**: Menghitung fluktuasi harga
- **Perhitungan Pajak**: Konversi harga sebelum/sesudah pajak

## Cara Menggunakan

1. Upload file Excel dengan format yang ditentukan
2. Pilih produk dan rentang tahun yang ingin dianalisis
3. Tentukan jumlah simulasi yang diinginkan
4. Klik tombol "Jalankan Simulasi"
5. Lihat hasil simulasi dan visualisasi data

## Format Data Excel

| Kolom             | Keterangan      | Contoh      |
| ----------------- | --------------- | ----------- |
| Tahun             | Tahun data      | 2021, 2022  |
| Bulan             | Bulan           | Jan, Feb    |
| Produk            | Nama produk     | Salsa 418ml |
| HARGA             | Harga produk    | 7.50, 8.20  |
| Tarif Pajak Total | Tarif pajak (%) | 11.0, 12.5  |

## Teknologi Yang Digunakan

- Python 3.x
- Streamlit (framework web)
- Pandas (analisis data)
- NumPy (komputasi numerik)
- Plotly (visualisasi data)

## Cara Menjalankan

1. Install dependensi:
   pip install streamlit pandas numpy plotly openpyxl

text

2. Jalankan aplikasi:
   streamlit run nama_file.py
