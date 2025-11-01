import pandas as pd
import numpy as np
import os
import sys

def load_iris_data(input_path):
    """
    Memuat dataset Iris dari file CSV
    
    Parameters:
    - input_path (str): Path ke file CSV
    
    Returns:
    - df (DataFrame): DataFrame yang berisi dataset Iris
    """
    try:
        df = pd.read_csv(input_path)
        print("✅ Dataset berhasil dimuat")
        return df
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error saat memuat dataset: {e}")
        sys.exit(1)

def exploratory_data_analysis(df):
    """
    Melakukan Exploratory Data Analysis (EDA) pada dataset
    
    Parameters:
    - df (DataFrame): DataFrame yang akan dianalisis
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    # 1. Informasi dasar dataset
    print("\n1. INFORMASI DATASET:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Kolom: {list(df.columns)}")
    
    # 2. Tampilkan 5 baris pertama
    print("\n2. 5 BARIS PERTAMA:")
    print(df.head())
    
    # 3. Info dataset
    print("\n3. INFO DATASET:")
    df.info()
    
    # 4. Statistik deskriptif
    print("\n4. STATISTIK DESKRIPTIF:")
    print(df.describe())
    
    # 5. Cek missing values
    print("\n5. MISSING VALUES:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # 6. Cek duplikat
    print("\n6. DUPLIKAT DATA:")
    duplicates = df.duplicated().sum()
    print(f"   - Jumlah duplikat: {duplicates}")
    
    # 7. Distribusi kelas target
    if 'Species' in df.columns:
        print("\n7. DISTRIBUSI SPECIES:")
        species_dist = df['Species'].value_counts()
        print(species_dist)

def preprocess_iris_data(df):
    """
    Melakukan preprocessing pada dataset Iris
    
    Parameters:
    - df (DataFrame): DataFrame mentah
    
    Returns:
    - df_processed (DataFrame): DataFrame yang sudah diproses
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Buat copy dataframe untuk menghindari warning
    df_processed = df.copy()
    
    # 1. Hapus kolom 'Id' jika ada (tidak diperlukan untuk modeling)
    if 'Id' in df_processed.columns:
        df_processed = df_processed.drop('Id', axis=1)
        print("✅ Kolom 'Id' dihapus")
    
    # 2. One-Hot Encoding untuk kolom 'Species'
    if 'Species' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['Species'], prefix='Species')
        print("✅ One-Hot Encoding untuk 'Species' selesai")
    
    # 3. Konversi boolean ke integer (opsional, untuk beberapa model ML)
    bool_columns = df_processed.select_dtypes(include='bool').columns
    if len(bool_columns) > 0:
        df_processed[bool_columns] = df_processed[bool_columns].astype(int)
        print("✅ Konversi boolean ke integer selesai")
    
    print(f"✅ Preprocessing selesai. Shape akhir: {df_processed.shape}")
    
    return df_processed

def save_processed_data(df_processed, output_path):
    """
    Menyimpan data yang sudah diproses
    
    Parameters:
    - df_processed (DataFrame): DataFrame yang sudah diproses
    - output_path (str): Path untuk menyimpan file hasil processing
    """
    try:
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Simpan dataframe
        df_processed.to_csv(output_path, index=False)
        print(f"✅ Data berhasil disimpan di: {output_path}")
        
        # Info file
        file_size = os.path.getsize(output_path) / 1024  # dalam KB
        print(f"   - Ukuran file: {file_size:.2f} KB")
        print(f"   - Shape data: {df_processed.shape}")
        
    except Exception as e:
        print(f"❌ Error saat menyimpan data: {e}")

def main():
    """
    Fungsi utama untuk menjalankan seluruh pipeline
    """
    print("🚀 MEMULAI AUTOMATED PREPROCESSING IRIS DATASET")
    print("="*60)
    
    # Resolve paths relative to this script so the script can be run from
    # different current working directories.
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Default locations (can be overridden with environment variables)
    input_path = os.environ.get('RAW_FILE_LOCATION', os.path.join(repo_root, 'namadataset_raw', 'Iris.csv'))
    output_path = os.environ.get('OUTPUT_PATH', os.path.join(script_dir, 'iris_preprocessing', 'iris_processed.csv'))
    
    # 1. Load data
    print(f"\n📥 Memuat data dari: {input_path}")
    df_raw = load_iris_data(input_path)
    
    # 2. Exploratory Data Analysis
    exploratory_data_analysis(df_raw)
    
    # 3. Preprocessing data
    df_processed = preprocess_iris_data(df_raw)
    
    # 4. Simpan data yang sudah diproses
    print(f"\n💾 Menyimpan data hasil processing ke: {output_path}")
    save_processed_data(df_processed, output_path)
    
    # 5. Tampilkan preview data hasil processing
    print("\n🔍 PREVIEW DATA HASIL PROCESSING:")
    print(df_processed.head())
    
    print("\n🎯 PIPEINE PREPROCESSING SELESAI!")
    print("Data siap untuk training model machine learning.")

if __name__ == "__main__":
    main()