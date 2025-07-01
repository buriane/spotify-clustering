import pandas as pd
import numpy as np


def generate_sample_data():
    """
    Generate sample Spotify user data for demo purposes.

    Returns:
        DataFrame: A pandas DataFrame with sample Spotify user data.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define possible genres
    genres = ['Pop', 'Rock', 'Hip-Hop', 'R&B', 'Jazz',
              'Electronic', 'Classical', 'Country', 'Folk', 'Metal']

    # Create sample data
    data = {
        'User_ID': range(1, 301),
        'Total_Waktu_Harian_Jam': np.random.uniform(0.5, 4.0, 300),
        'Durasi_Sesi_Rata_rata_Jam': np.random.uniform(0.2, 1.5, 300),
        'Frekuensi_Akses_Harian': np.random.randint(1, 15, 300),
        'Total_Lagu_Harian': np.random.randint(10, 150, 300),
        'Genre_Sering_Didengarkan': [', '.join(np.random.choice(
            genres,
            size=np.random.randint(1, 4),
            replace=False
        )) for _ in range(300)]
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Generate the sample data
    sample_df = generate_sample_data()

    # Save to CSV
    sample_df.to_csv('contoh_dataset_spotify.csv', index=False)
    print("Sample dataset created: contoh_dataset_spotify.csv")
