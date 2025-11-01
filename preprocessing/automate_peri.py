import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

def preprocess_iris_data(raw_file_path: str, output_dir: str):
    """
    Melakukan preprocessing data Iris secara otomatis, termasuk:
    1. Memuat data.
    2. Menghapus kolom 'Id'.
    3. Melakukan One-Hot Encoding pada kolom 'Species'.
    4. Memisahkan data menjadi fitur (X) dan target (y).
    5. Menyimpan data yang sudah diproses.
    
    Args:
        raw_file_path (str): Path lengkap menuju file CSV mentah (Iris.csv).
        output_dir (str): Folder untuk menyimpan file CSV hasil preprocessing.
    """
    # 1. Muat Data
    print(f"Memuat data dari: {raw_file_path}")
    df = pd.read_csv(raw_file_path)

    # 2. Hapus Kolom 'Id' (Sesuai praktik umum)
    df = df.drop('Id', axis=1)

    # 3. Definisikan Kolom Target dan Fitur
    # Kolom fitur adalah semua kolom kecuali 'Species'
    X = df.drop('Species', axis=1)
    # Kolom target adalah 'Species' (yang akan di-encode)
    y = df['Species'] 

    # --- Konversi Tahapan Manual ke Struktur Otomatis (ColumnTransformer & Pipeline) ---
    
    # Karena di notebook hanya ada One-Hot Encoding, kita hanya perlu satu transformer
    
    # Tentukan fitur yang akan di-transformasi (hanya kolom kategorikal 'Species')
    # Catatan: Kita memproses target 'Species' di sini untuk menghasilkan kolom biner
    
    # Definisikan transformer untuk One-Hot Encoding
    ohe_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Gunakan ColumnTransformer untuk menerapkan OHE ke kolom 'Species'
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', ohe_transformer, ['Species'])
        ],
        # Passthrough/drop tidak diperlukan di sini karena kita hanya memproses 'Species'
        # dan akan memisahkannya menjadi target y
        remainder='drop' 
    )
    
    # --- Terapkan Transformasi pada Target (y) ---
    print("Menerapkan One-Hot Encoding pada target 'Species'...")
    y_encoded = preprocessor.fit_transform(y.to_frame())
    
    # Buat DataFrame dari hasil encoding untuk kolom target (y)
    # Gunakan nama kolom yang dihasilkan oleh OneHotEncoder
    target_names = preprocessor.get_feature_names_out(['Species'])
    y_encoded_df = pd.DataFrame(y_encoded, columns=target_names)
    
    # Gabungkan Fitur (X) dan Target (y_encoded) yang sudah siap dilatih
    X.reset_index(drop=True, inplace=True)
    final_df = pd.concat([X, y_encoded_df], axis=1)
    
    print("Preprocessing selesai. Data siap dilatih.")
    
    # --- Simpan Data Hasil Preprocessing ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'iris_processed_data.csv')
    
    final_df.to_csv(output_path, index=False)
    print(f"\nData yang diproses disimpan di: {output_path}")
    print(f"Shape data akhir: {final_df.shape}")

# --- Bagian Main (Untuk menjalankan script dan menyimpan hasilnya) ---
if __name__ == "__main__":
    # Sesuaikan path ini dengan lokasi file Anda di sistem lokal/GitHub
    # Contoh path relatif (mengikuti struktur repository)
    
    # CATATAN: Ganti 'Nama-siswa' dengan nama Anda yang sebenarnya!
    # Lokasi file mentah
    RAW_FILE_LOCATION = '../namadataset_raw/Iris.csv'
    
    # Lokasi folder output hasil preprocessing
    OUTPUT_FOLDER = '../preprocessing/iris_preprocessing' 

    # Panggil fungsi utama
    # Pastikan file Iris.csv sudah ada di '../namadataset_raw/'
    preprocess_iris_data(RAW_FILE_LOCATION, OUTPUT_FOLDER)