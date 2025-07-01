import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io
import base64
import warnings
import os.path
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Analisis dan Clustering Pengguna Spotify",
    page_icon="🎵",
    layout="wide"
)

# Title and description
st.title("🎵 Analisis dan Clustering Pengguna Spotify")
st.markdown("""
Aplikasi ini melakukan analisis clustering pada data penggunaan Spotify untuk mengidentifikasi segmen pengguna 
dan menemukan pola dalam preferensi musik. Unggah file CSV data pengguna Spotify Anda untuk memulai!
""")

# Sidebar
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.info(
        """
        Aplikasi ini menganalisis data perilaku pengguna Spotify untuk mengidentifikasi segmen pengguna 
        yang berbeda dan menemukan aturan asosiasi antara genre musik.
        
        **Fitur:**
        - Segmentasi pengguna dengan K-means clustering
        - Principal Component Analysis (PCA)
        - Penambangan aturan asosiasi untuk preferensi genre
        - Visualisasi dan wawasan yang detail
        
        *Proyek ini dibuat sebagai tugas akhir mata kuliah Machine Learning*
        """
    )

    st.header("Petunjuk")
    st.info(
        """
        1. Unduh file CSV sampel atau siapkan data Anda sendiri
        2. Unggah file CSV Anda menggunakan uploader
        3. Jelajahi hasil analisis
        4. Klik 'Reset' untuk menganalisis dataset yang berbeda
        """
    )

# Function to generate a download link for the sample data


def generate_sample_data():
    # Create sample data similar to the one in the code
    data = {
        'User_ID': range(1, 301),
        'Total_Waktu_Harian_Jam': np.random.uniform(0.5, 4.0, 300),
        'Durasi_Sesi_Rata_rata_Jam': np.random.uniform(0.2, 1.5, 300),
        'Frekuensi_Akses_Harian': np.random.randint(1, 15, 300),
        'Total_Lagu_Harian': np.random.randint(10, 150, 300),
        'Genre_Sering_Didengarkan': [', '.join(np.random.choice(
            ['Pop', 'Rock', 'Hip-Hop', 'R&B', 'Jazz', 'Electronic',
                'Classical', 'Country', 'Folk', 'Metal'],
            size=np.random.randint(1, 4),
            replace=False
        )) for _ in range(300)]
    }

    sample_df = pd.DataFrame(data)
    return sample_df


def get_custom_data():
    # Path to custom data CSV file
    custom_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'Spotify_Usage_Data_301.csv'
    )

    # Read the custom data if it exists, otherwise return an empty DataFrame
    if os.path.exists(custom_data_path):
        return pd.read_csv(custom_data_path)
    else:
        st.error(
            "Custom data file not found. Please make sure 'Spotify_Usage_Data_301.csv' exists in the application folder.")
        return pd.DataFrame()


def create_download_link(data_df, filename, button_text):
    csv = data_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return href


# Dataset download section
st.subheader("Opsi Unduh Dataset")
col1, col2 = st.columns(2)

# Create sample data download button
with col1:
    if st.button('Buat Dataset Sampel Acak'):
        sample_df = generate_sample_data()
        st.markdown(create_download_link(
            sample_df, 'contoh_dataset_spotify.csv', 'Unduh CSV Sampel Acak'),
            unsafe_allow_html=True)
        st.success(
            'Dataset sampel acak telah dibuat! Klik tautan di atas untuk mengunduh.')

# Create custom data download button
with col2:
    if st.button('Unduh Dataset Spotify'):
        custom_df = get_custom_data()
        if not custom_df.empty:
            st.markdown(create_download_link(
                custom_df, 'Spotify_Usage_Data_301.csv', 'Unduh CSV'),
                unsafe_allow_html=True)
            st.success(
                'Dataset siap! Klik tautan di atas untuk mengunduh.')
        else:
            st.error(
                'Tidak dapat memuat dataset. Pastikan file ada di folder aplikasi.')

st.markdown("---")  # Horizontal line for visual separation

# File uploader
uploaded_file = st.file_uploader(
    "Unggah data penggunaan Spotify Anda (CSV)", type="csv")

# Reset button
if st.button('Reset'):
    st.experimental_rerun()

# Main analysis workflow
if uploaded_file is not None:
    # Display progress
    progress_bar = st.progress(0)

    # Create tabs for organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🧩 PCA & Clustering",
        "👥 Profil Cluster",
        "🔗 Aturan Asosiasi",
        "📝 Kesimpulan"
    ])

    # Load and preprocess data
    with st.spinner("Memuat dan memproses data..."):
        # Load data
        df = pd.read_csv(uploaded_file)
        progress_bar.progress(10)

        # PREPROCESSING DATA
        with tab1:
            st.header("Data Overview")

            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    f"Dataset loaded: {df.shape[0]} users, {df.shape[1]} features")
            with col2:
                # Check for missing values
                missing_values = df.isnull().sum().sum()
                st.write(f"Missing values: {missing_values}")

            # Handle missing values if any
            if missing_values > 0:
                df = df.fillna(df.select_dtypes(include=[np.number]).mean())
                st.info("Missing values were filled with mean values")

            # Display sample data
            st.subheader("Sampel Data")
            st.dataframe(df.head())

            # Prepare genre dummies
            genre_dummies = df['Genre_Sering_Didengarkan'].str.get_dummies(
                sep=', ')
            st.subheader("Distribusi Genre")
            st.write(f"Ditemukan {genre_dummies.shape[1]} genre unik")

            # Show genres distribution
            genres_flat = []
            for genres_str in df['Genre_Sering_Didengarkan']:
                genres = [g.strip() for g in genres_str.split(',')]
                genres_flat.extend(genres)

            genre_counts = pd.Series(genres_flat).value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            genre_counts.head(10).plot(kind='bar', ax=ax)
            plt.title('10 Genre Terpopuler')
            plt.xlabel('Genre')
            plt.ylabel('Jumlah')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        # Prepare numerical features
        numerical_features = ['Total_Waktu_Harian_Jam', 'Durasi_Sesi_Rata_rata_Jam',
                              'Frekuensi_Akses_Harian', 'Total_Lagu_Harian']
        X_numerical = df[numerical_features].copy()

        # Combine features
        X_combined = pd.concat([X_numerical, genre_dummies], axis=1)

        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        progress_bar.progress(20)

    # PRINCIPAL COMPONENT ANALYSIS (PCA)
    with st.spinner("Melakukan Analisis Komponen Utama (PCA)..."):
        # Apply PCA
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)

        # Create PCA dataframe
        pca_2d_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])

        # Display PCA results
        explained_var = pca_2d.explained_variance_ratio_

        with tab2:
            st.header("Analisis Komponen Utama (PCA)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Varians PC1",
                          f"{explained_var[0]*100:.2f}%")
            with col2:
                st.metric("Varians PC2",
                          f"{explained_var[1]*100:.2f}%")
            with col3:
                st.metric("Total Varians",
                          f"{sum(explained_var)*100:.2f}%")

            # Visualize PCA
            st.subheader("Distribusi Data dalam Ruang PCA")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       alpha=0.6, c='skyblue', edgecolors='navy')
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% varians)')
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% varians)')
            ax.set_title('Distribusi Data dalam Ruang PCA')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        progress_bar.progress(30)

    # DETERMINE OPTIMAL NUMBER OF CLUSTERS
    with st.spinner("Menentukan jumlah cluster optimal..."):
        k_range = range(2, 8)
        inertias = []
        silhouette_scores = []

        with tab2:
            st.subheader("Menentukan Jumlah Cluster Optimal")

            # Create placeholder for cluster metrics
            cluster_metrics_placeholder = st.empty()

            # Show metrics calculation in progress
            cluster_metrics_data = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca_2d)

                inertias.append(kmeans.inertia_)
                sil_score = silhouette_score(X_pca_2d, cluster_labels)
                silhouette_scores.append(sil_score)

                cluster_metrics_data.append(
                    {"k": k, "Silhouette Score": f"{sil_score:.3f}"})

            # Show metrics table
            cluster_metrics_placeholder.dataframe(
                pd.DataFrame(cluster_metrics_data))

            # Find optimal k
            optimal_k = k_range[np.argmax(silhouette_scores)]
            st.success(
                f"Jumlah cluster optimal: {optimal_k} (Silhouette Score tertinggi)")

            # Visualize cluster evaluation
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Elbow Method")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col2:
                st.subheader("Silhouette Score")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(k_range, silhouette_scores,
                        'go-', linewidth=2, markersize=8)
                ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
                           label=f'Optimal k={optimal_k}')
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Score')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        progress_bar.progress(50)

    # FINAL CLUSTERING
    with st.spinner("Melakukan clustering akhir..."):
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_clusters = final_kmeans.fit_predict(X_pca_2d)

        # Add cluster labels
        pca_2d_df['Cluster'] = final_clusters
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = final_clusters

        with tab2:
            st.subheader("Hasil K-Means Clustering")

            # Display cluster distribution
            st.write("Distribusi Cluster:")
            cluster_counts = pd.Series(
                final_clusters).value_counts().sort_index()
            cluster_dist_data = []

            for cluster_id, count in cluster_counts.items():
                cluster_dist_data.append({
                    "Cluster": cluster_id,
                    "Count": count,
                    "Percentage": f"{count/len(df)*100:.1f}%"
                })

            st.dataframe(pd.DataFrame(cluster_dist_data))

            # Visualization of clusters
            st.subheader("Visualisasi Cluster")

            # Plot clusters
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))

            for i in range(optimal_k):
                cluster_points = pca_2d_df[pca_2d_df['Cluster'] == i]
                ax.scatter(cluster_points['PC1'], cluster_points['PC2'],
                           c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=60)

            centroids_pca = final_kmeans.cluster_centers_
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                       c='black', marker='x', s=200, linewidths=3, label='Centroid')

            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
            ax.set_title('K-Means Clustering Results')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Cluster characteristics visualization
            st.subheader("Cluster Characteristics")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            metrics = [
                ('Total_Waktu_Harian_Jam', 'Daily Listening Time (Hours)', 'h'),
                ('Durasi_Sesi_Rata_rata_Jam',
                 'Average Session Duration (Hours)', 'h'),
                ('Total_Lagu_Harian', 'Daily Songs Count', ''),
                ('Frekuensi_Akses_Harian', 'Daily Access Frequency', 'x')
            ]

            for idx, (col, title, unit) in enumerate(metrics):
                ax = axes[idx]
                values = df_with_clusters.groupby('Cluster')[col].mean().values

                bars = ax.bar(range(optimal_k), values,
                              color=colors[:optimal_k])
                ax.set_xlabel('Cluster')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.set_xticks(range(optimal_k))

                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                            f'{values[i]:.1f}{unit}', ha='center', va='bottom')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        progress_bar.progress(70)

    # CLUSTER PROFILE ANALYSIS
    with st.spinner("Menganalisis profil cluster..."):
        def analyze_genres_in_cluster(cluster_df):
            """Analyze genre preferences in a cluster"""
            genres_list = []
            for genres_str in cluster_df['Genre_Sering_Didengarkan']:
                genres = [g.strip() for g in genres_str.split(',')]
                genres_list.extend(genres)
            return pd.Series(genres_list).value_counts()

        cluster_profiles = {}
        summary_data = []

        with tab3:
            st.header("Analisis Profil Cluster")

            for cluster_id in range(optimal_k):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]

                # Basic statistics
                stats = {
                    'waktu_harian': cluster_data['Total_Waktu_Harian_Jam'].mean(),
                    'durasi_sesi': cluster_data['Durasi_Sesi_Rata_rata_Jam'].mean(),
                    'lagu_harian': cluster_data['Total_Lagu_Harian'].mean(),
                    'frekuensi_akses': cluster_data['Frekuensi_Akses_Harian'].mean()
                }

                # Genre analysis
                genre_counts = analyze_genres_in_cluster(cluster_data)

                # Determine cluster label based on characteristics
                if stats['waktu_harian'] > 2:
                    intensity = "Heavy"
                elif stats['waktu_harian'] > 1:
                    intensity = "Moderate"
                else:
                    intensity = "Light"

                top_genre = genre_counts.index[0] if len(
                    genre_counts) > 0 else 'Mixed'
                cluster_label = f"{intensity} {top_genre} Listeners"

                # Store for summary
                cluster_profiles[cluster_id] = {
                    'size': len(cluster_data),
                    'stats': stats,
                    'top_genres': genre_counts.head(3).to_dict(),
                    'label': cluster_label
                }

                summary_data.append({
                    'Cluster': cluster_id,
                    'Label': cluster_label,
                    'Users': len(cluster_data),
                    'Percentage': f"{len(cluster_data)/len(df)*100:.1f}%",
                    'Daily Time (hrs)': f"{stats['waktu_harian']:.2f}",
                    'Favorite Genre': top_genre
                })

                # Create an expander for each cluster
                with st.expander(f"Cluster {cluster_id}: {cluster_label} ({len(cluster_data)} users)"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Metrik Utama")
                        st.write(
                            f"Waktu mendengarkan harian: {stats['waktu_harian']:.2f} jam")
                        st.write(
                            f"Durasi sesi rata-rata: {stats['durasi_sesi']:.2f} jam")
                        st.write(
                            f"Jumlah lagu harian: {stats['lagu_harian']:.1f} lagu")
                        st.write(
                            f"Frekuensi akses harian: {stats['frekuensi_akses']:.1f} kali/hari")

                    with col2:
                        st.subheader("Genre Teratas")
                        top_genres_df = pd.DataFrame({
                            'Genre': genre_counts.head(5).index,
                            'Count': genre_counts.head(5).values,
                            'Percentage': [(count / len(cluster_data)) * 100 for count in genre_counts.head(5)]
                        })
                        top_genres_df['Percentage'] = top_genres_df['Percentage'].apply(
                            lambda x: f"{x:.1f}%")
                        st.dataframe(top_genres_df)

                    # Visualize genre distribution for this cluster
                    st.subheader("Distribusi Genre")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    genre_counts.head(7).plot(kind='bar', ax=ax)
                    plt.title(f'Genre Teratas untuk Cluster {cluster_id}')
                    plt.ylabel('Jumlah')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

            # Summary table of all clusters
            st.subheader("Ringkasan Cluster")
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

            # Visualization of cluster sizes
            st.subheader("Ukuran Cluster")
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster_sizes = [
                len(df_with_clusters[df_with_clusters['Cluster'] == i]) for i in range(optimal_k)]
            labels = [
                f"{i}: {cluster_profiles[i]['label']}" for i in range(optimal_k)]
            ax.pie(cluster_sizes, labels=labels, autopct='%1.1f%%',
                   startangle=90, colors=colors[:optimal_k])
            ax.axis('equal')
            plt.title('Cluster Distribution')
            st.pyplot(fig)

        progress_bar.progress(80)

    # ASSOCIATION RULE MINING
    with st.spinner("Melakukan penambangan aturan asosiasi..."):
        def prepare_genre_transactions(cluster_df):
            """Convert genre strings to transaction format"""
            transactions = []
            for genres_str in cluster_df['Genre_Sering_Didengarkan']:
                genres = [genre.strip() for genre in genres_str.split(',')]
                transactions.append(genres)
            return transactions

        def mine_association_rules(df_encoded, min_support=0.1, min_confidence=0.2):
            """Mine association rules"""
            try:
                frequent_itemsets = apriori(
                    df_encoded, min_support=min_support, use_colnames=True)
                if len(frequent_itemsets) == 0:
                    return pd.DataFrame()

                rules = association_rules(
                    frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                return rules.sort_values('confidence', ascending=False) if len(rules) > 0 else pd.DataFrame()
            except:
                return pd.DataFrame()

        def format_itemset(itemset):
            """Format itemset for display"""
            return ", ".join(sorted(list(itemset))) if isinstance(itemset, frozenset) else str(itemset)

        cluster_association_rules = {}
        association_summary = []

        with tab4:
            st.header("Penambangan Aturan Asosiasi")
            st.write(
                "Menganalisis preferensi genre dan menemukan aturan asosiasi...")

            for cluster_id in range(optimal_k):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]

                with st.expander(f"Association Rules for Cluster {cluster_id} ({cluster_profiles[cluster_id]['label']})"):
                    st.write(f"Analyzing {len(cluster_data)} users")

                    if len(cluster_data) < 5:
                        st.warning(
                            "Data tidak cukup untuk penambangan aturan asosiasi")
                        cluster_association_rules[cluster_id] = pd.DataFrame()
                        continue

                    # Prepare transactions
                    transactions = prepare_genre_transactions(cluster_data)

                    # Create one-hot encoded matrix
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                    if len(df_encoded.columns) < 2:
                        st.warning(
                            "Variasi genre tidak cukup untuk aturan asosiasi")
                        cluster_association_rules[cluster_id] = pd.DataFrame()
                        continue

                    # Adjust min_support based on cluster size
                    min_support = max(0.05, 2/len(cluster_data))

                    # Mine rules
                    rules = mine_association_rules(
                        df_encoded, min_support=min_support, min_confidence=0.2)
                    cluster_association_rules[cluster_id] = rules

                    if len(rules) == 0:
                        st.info("Tidak ditemukan aturan asosiasi yang signifikan")
                        association_summary.append({
                            'Cluster': cluster_id,
                            'Rules Count': 0,
                            'Top Rule': 'Tidak ada',
                            'Confidence': 'N/A'
                        })
                    else:
                        st.success(f"Ditemukan {len(rules)} aturan asosiasi")

                        # Show rules table
                        rules_display = rules.copy()
                        rules_display['antecedents'] = rules_display['antecedents'].apply(
                            format_itemset)
                        rules_display['consequents'] = rules_display['consequents'].apply(
                            format_itemset)
                        rules_display = rules_display[[
                            'antecedents', 'consequents', 'support', 'confidence', 'lift']]
                        rules_display = rules_display.rename(columns={
                            'antecedents': 'Jika mendengarkan',
                            'consequents': 'Kemungkinan mendengarkan',
                            'support': 'Support',
                            'confidence': 'Confidence',
                            'lift': 'Lift'
                        })

                        # Show top rules
                        st.subheader("Aturan Asosiasi Teratas")
                        st.dataframe(rules_display.head(5))

                        # Add to summary
                        top_rule = rules.iloc[0]
                        top_rule_str = f"{format_itemset(top_rule['antecedents'])} → {format_itemset(top_rule['consequents'])}"
                        association_summary.append({
                            'Cluster': cluster_id,
                            'Rules Count': len(rules),
                            'Top Rule': top_rule_str[:40] + "..." if len(top_rule_str) > 40 else top_rule_str,
                            'Confidence': f"{top_rule['confidence']:.3f}"
                        })

            # Association rules summary
            st.subheader("Ringkasan Aturan Asosiasi")
            assoc_summary_df = pd.DataFrame(association_summary)
            st.dataframe(assoc_summary_df)

            # Visualize association rules
            st.subheader("Visualisasi Aturan Asosiasi")

            # Count rules per cluster
            rules_counts = [len(cluster_association_rules[i])
                            for i in range(optimal_k)]

            col1, col2 = st.columns(2)

            with col1:
                # Number of rules per cluster
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(range(optimal_k), rules_counts,
                              color=colors[:optimal_k])
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Number of Association Rules')
                ax.set_title('Number of Association Rules per Cluster')
                ax.set_xticks(range(optimal_k))
                for i, count in enumerate(rules_counts):
                    ax.text(i, count + 0.1, str(count),
                            ha='center', va='bottom')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col2:
                # Average confidence per cluster
                avg_confidences = []
                for cluster_id in range(optimal_k):
                    rules = cluster_association_rules[cluster_id]
                    avg_confidences.append(
                        rules['confidence'].mean() if len(rules) > 0 else 0)

                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(range(optimal_k), avg_confidences,
                              color=colors[:optimal_k])
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Average Confidence')
                ax.set_title('Average Rule Confidence per Cluster')
                ax.set_xticks(range(optimal_k))
                for i, conf in enumerate(avg_confidences):
                    if conf > 0:
                        ax.text(i, conf + 0.01,
                                f'{conf:.2f}', ha='center', va='bottom')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        progress_bar.progress(90)

    # CONCLUSIONS / KESIMPULAN
    with st.spinner("Menghasilkan kesimpulan..."):
        with tab5:
            st.header("Kesimpulan Analisis")

            st.subheader("Hasil Clustering")
            st.write(f"• Jumlah cluster optimal: {optimal_k}")
            st.write(f"• Total pengguna yang dianalisis: {len(df)}")
            st.write(
                f"• Varians yang dijelaskan oleh PCA: {sum(explained_var)*100:.1f}%")

            st.subheader("Distribusi Pengguna")
            for cluster_id in range(optimal_k):
                profile = cluster_profiles[cluster_id]
                st.write(
                    f"• Cluster {cluster_id} ({profile['label']}): {profile['size']} pengguna ({profile['size']/len(df)*100:.1f}%)")

            st.subheader("Aturan Asosiasi")
            total_rules = sum(
                len(cluster_association_rules[i]) for i in range(optimal_k))
            clusters_with_rules = sum(1 for i in range(
                optimal_k) if len(cluster_association_rules[i]) > 0)
            st.write(f"• Total aturan asosiasi yang ditemukan: {total_rules}")
            st.write(
                f"• Cluster dengan aturan signifikan: {clusters_with_rules} dari {optimal_k}")

            st.subheader("Wawasan Bisnis")
            st.write("Berdasarkan analisis, berikut beberapa wawasan bisnis utama:")

            for cluster_id in range(optimal_k):
                profile = cluster_profiles[cluster_id]
                # Keep the label in English since it's used throughout the app
                st.write(f"**{profile['label']} (Cluster {cluster_id}):**")

                # Generate insights based on cluster characteristics
                if profile['stats']['waktu_harian'] > 2:
                    st.write(
                        "• Pengguna dengan keterlibatan tinggi yang menghabiskan waktu signifikan di platform")
                    st.write("• Peluang untuk konversi langganan premium")
                elif profile['stats']['waktu_harian'] < 1:
                    st.write(
                        "• Pengguna dengan keterlibatan rendah - target untuk kampanye re-engagement")
                    st.write(
                        "• Peluang untuk meningkatkan durasi sesi dengan konten yang dipersonalisasi")

                # Generate genre-specific recommendations
                top_genre = list(profile['top_genres'].keys())[
                    0] if profile['top_genres'] else "genre campuran"
                st.write(
                    f"• Minat utama pada {top_genre} - peluang untuk playlist dan promosi artis yang ditargetkan")

                # Association rule insights
                if cluster_id in cluster_association_rules and len(cluster_association_rules[cluster_id]) > 0:
                    st.write(
                        "• Asosiasi genre yang kuat terdeteksi - peluang untuk promosi silang antar genre")

                st.write("")

            st.success(
                "Analisis selesai! Anda sekarang dapat mengunduh hasil atau me-reset aplikasi untuk menganalisis dataset yang berbeda.")

        progress_bar.progress(100)

# Download link for custom dataset
if uploaded_file is not None:
    @st.cache_resource
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df_with_clusters)

    st.download_button(
        label="Unduh Hasil Analisis (CSV)",
        data=csv_data,
        file_name='spotify_analysis_results.csv',
        mime='text/csv',
        key='download-csv'
    )

# Footer
st.markdown("""
---
*Aplikasi ini dikembangkan sebagai Tugas Akhir mata kuliah Machine Learning*
""")
