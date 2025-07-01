# Spotify User Clustering & Analysis App

This Streamlit application performs cluster analysis on Spotify usage data to identify user segments and discover patterns in music preferences.

## Features

- Upload and analyze your own Spotify usage data (CSV format)
- Visualize user segmentation with K-means clustering
- Apply Principal Component Analysis (PCA) for dimensionality reduction
- Generate association rules to discover relationships between music genres
- Interactive visualizations and detailed insights

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
