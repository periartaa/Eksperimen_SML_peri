import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

# --- Fungsi Utility (Sama seperti modelling.py) ---

def load_processed_data(input_path):
    """
    Memuat dataset yang sudah diproses dari file CSV.
    """
    try:
        df = pd.read_csv(input_path)
        print(" Dataset processed berhasil dimuat")
        return df
    except FileNotFoundError:
        print(f" File tidak ditemukan: {input_path}. Pastikan langkah preprocessing sudah dijalankan.")
        sys.exit(1)
    except Exception as e:
        print(f" Error saat memuat dataset: {e}")
        sys.exit(1)

def train_and_log_tuned_model(df):
    """
    Melatih model Klasifikasi (Logistic Regression) dengan Hyperparameter Tuning
    dan menggunakan MANUAL LOGGING.
    """
    print("\n" + "="*70)
    print("MODEL TUNING DENGAN MANUAL LOGGING DAN HYPERPARAMETER TUNING")
    print("="*70)
    
    # 1. Persiapan Data (Sama seperti sebelumnya)
    target_columns = [col for col in df.columns if col.startswith('Species_')]
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    if not target_columns:
        print(" Kolom target 'Species_' tidak ditemukan.")
        return

    X = df[feature_columns]
    y_ohe = df[target_columns]
    
    # Konversi OHE ke Label Tunggal (1D)
    y = y_ohe.idxmax(axis=1) 
    
    # Pisahkan data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train (1D): {y_train.shape}")
    
    # 2. DEFINISI HYPERPARAMETER TUNING
    
    # JANGAN AKTIFKAN AUTOLOG (karena diminta manual logging)
    # mlflow.sklearn.autolog(disable=True) 

    experiment_name = "Iris_Classification_Skilled_Tuning"
    mlflow.set_experiment(experiment_name)
    
    # Hyperparameter Grid untuk Logistic Regression
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000] 
    }
    
    base_model = LogisticRegression(random_state=42)
    
    # GridSearchCV untuk mencari kombinasi terbaik
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy'
    )
    
    # 3. MEMULAI RUN MLFLOW DENGAN MANUAL LOGGING
    
    with mlflow.start_run() as run:
        
        # Log Tag/Informasi Run
        mlflow.set_tag("model_type", "Logistic Regression with Tuning")
        mlflow.set_tag("data_split", "70-30")
        mlflow.set_tag("tuning_method", "GridSearchCV")
        
        # Log Hyperparameter Tuning Strategy (seperti autolog)
        mlflow.log_param("tuning_cv", 5)
        mlflow.log_param("tuning_scoring", 'accuracy')

        print("\n Memulai GridSearchCV...")
        grid_search.fit(X_train, y_train)
        print(" GridSearchCV Selesai.")
        
        # Ambil Model Terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # 4. MANUAL LOGGING HASIL MODEL TERBAIK
        
        # Log Hyperparameter Terbaik
        print("\nManually Logging Parameters...")
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        mlflow.log_param("model_class", type(best_model).__name__)
        
        # Log Metrik pada Data Uji
        y_pred = best_model.predict(X_test)
        
        # Hitung Metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log Metrik (sama dengan yang dicatat autolog)
        print("Manually Logging Metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        
        # 5. MANUAL LOGGING MODEL ARTEFAK
        
        print("Manually Logging Model Artefact...")
        mlflow.sklearn.log_model(
            best_model, 
            "model_best_tuned", 
            registered_model_name="IrisLogisticRegressionTuned"
        )
        
        print("\n Proses Tuning dan Manual Logging Selesai.")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Best Parameters: {best_params}")
        print(f"Accuracy on Test Set: {accuracy:.4f}")
        
    print("\n Model Training Selesai! Cek MLflow Tracking UI.")

def main():
    """
    Fungsi utama untuk menjalankan pipeline modeling.
    """
    
    # Ganti dengan PATH LOKASI SEBENARNYA dari data processed Anda
    processed_file_path = r'C:\Peri\DICODING\ACCOUNT StudentPro\Eksperimen_SML_periart\preprocessing\iris_preprocessing\iris_processed.csv' 
    
    # 1. Load data
    df_processed = load_processed_data(processed_file_path)
    
    # 2. Train model dengan Manual Log dan Tuning
    train_and_log_tuned_model(df_processed)

if __name__ == "__main__":
    main()