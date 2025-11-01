import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_iris_data(raw_file_path, output_folder):
    """
    Preprocess Iris dataset
    """
    try:
        # Cek apakah file exists
        if not os.path.exists(raw_file_path):
            # Coba path alternatif
            alt_path = os.path.join(os.getcwd(), 'namadataset_raw', 'Iris.csv')
            if os.path.exists(alt_path):
                raw_file_path = alt_path
                print(f"File ditemukan di: {alt_path}")
            else:
                raise FileNotFoundError(f"File tidak ditemukan: {raw_file_path}")
        
        print(f"Memuat data dari: {raw_file_path}")
        
        # Load dataset
        df = pd.read_csv(raw_file_path)
        print(f"Data berhasil dimuat. Shape: {df.shape}")
        
        # Pastikan output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Data cleaning
        print("Melakukan data cleaning...")
        
        # Hapus duplikat
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        print(f"Duplikat dihapus: {initial_count - final_count} baris")
        
        # Handle missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Missing values ditemukan: {missing_values}")
            # Untuk numerical columns, fill dengan median
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # Untuk categorical columns, fill dengan mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        # Feature engineering
        print("Melakukan feature engineering...")
        
        # Jika ada kolom species, encode
        if 'species' in df.columns:
            le = LabelEncoder()
            df['species_encoded'] = le.fit_transform(df['species'])
            print(f"Species encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Normalisasi numerical features
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
        # Exclude target variable jika ada
        if 'species_encoded' in numerical_features:
            numerical_features = numerical_features.drop('species_encoded')
        
        if len(numerical_features) > 0:
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            print(f"Features dinormalisasi: {list(numerical_features)}")
        
        # Simpan processed data
        output_path = os.path.join(output_folder, 'iris_processed_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Data berhasil diproses dan disimpan di: {output_path}")
        print(f"Final data shape: {df.shape}")
        print("\nData preview:")
        print(df.head())
        print("\nData info:")
        print(df.info())
        
        return df
        
    except Exception as e:
        print(f"Error selama preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Gunakan path relatif yang lebih reliable
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(current_dir)
    
    # Define paths
    RAW_FILE_PATH = os.path.join(repo_root, 'namadataset_raw', 'Iris.csv')
    OUTPUT_FOLDER = os.path.join(current_dir, 'iris_preprocessing')
    
    print(f"Current directory: {current_dir}")
    print(f"Repo root: {repo_root}")
    print(f"Raw file path: {RAW_FILE_PATH}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Jalankan preprocessing
    preprocess_iris_data(RAW_FILE_PATH, OUTPUT_FOLDER)