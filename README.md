# Analisis dan Clustering Pengguna Spotify

Aplikasi Streamlit ini melakukan analisis clustering pada data penggunaan Spotify untuk mengidentifikasi segmen pengguna dan menemukan pola dalam preferensi musik.

## Fitur

- Unggah dan analisis data penggunaan Spotify Anda (format CSV)
- Visualisasikan segmentasi pengguna dengan clustering K-means
- Terapkan Principal Component Analysis (PCA) untuk reduksi dimensi
- Hasilkan aturan asosiasi untuk menemukan hubungan antara genre musik
- Visualisasi interaktif dan wawasan mendetail
- Unduh dataset contoh atau dataset yang dihasilkan secara acak

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install using `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - mlxtend

### Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

### Using the App

1. When the app starts, you can download a sample dataset by clicking "Generate Sample Dataset"
2. Upload your CSV file using the file uploader
3. The app will process the data and display the results in organized tabs:
   - Data Overview: Basic statistics and distributions
   - PCA & Clustering: Dimensionality reduction and cluster visualization
   - Cluster Profiles: Detailed analysis of each user segment
   - Association Rules: Relationships between music genres
   - Conclusions: Summary insights and business recommendations

## Data Format

Your CSV file should contain the following columns:

- `User_ID`: Unique identifier for each user
- `Total_Waktu_Harian_Jam`: Average daily listening time in hours
- `Durasi_Sesi_Rata_rata_Jam`: Average session duration in hours
- `Frekuensi_Akses_Harian`: Average daily access frequency
- `Total_Lagu_Harian`: Average number of songs played daily
- `Genre_Sering_Didengarkan`: Comma-separated list of frequently listened genres

## Deployment ke Streamlit Cloud

Aplikasi ini siap untuk di-deploy ke Streamlit Cloud dengan langkah-langkah berikut:

1. Buat akun di [Streamlit Cloud](https://streamlit.io/cloud)
2. Hubungkan dengan repositori GitHub Anda yang berisi aplikasi ini
3. Konfigurasi deployment:
   - Set file utama sebagai `app.py`
   - Tentukan versi Python (3.10 direkomendasikan)
   - Pastikan semua persyaratan dependensi ada dalam `requirements.txt`

### File Konfigurasi

Aplikasi ini menggunakan file konfigurasi berikut untuk deployment:

1. `.streamlit/config.toml` - Berisi konfigurasi tema dan pengaturan server
2. `.streamlit/secrets.toml` - Berisi kredensial dan konfigurasi sensitif (tidak dicommit ke repository)
3. `runtime.txt` - Menentukan versi Python yang digunakan

### Secrets Management

Jika aplikasi perlu mengakses layanan eksternal atau database:

1. Tambahkan kredensial yang diperlukan ke `.streamlit/secrets.toml` secara lokal
2. Tambahkan secrets yang sama melalui dashboard Streamlit Cloud (Advanced Settings > Secrets)

### Custom Domain

Untuk menggunakan domain kustom (contoh: spotifycluster.id):

1. Daftarkan domain yang diinginkan melalui registrar domain
2. Konfigurasikan domain di dashboard Streamlit Cloud
3. Ikuti instruksi untuk menyiapkan DNS dan SSL