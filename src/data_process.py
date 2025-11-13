# src/data_process.py

import pandas as pd
import os

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    """
    Lecture, nettoyage et préparation des données.
    Sauvegarde le dataset préparé dans output_dir.
    """
    # Créer le répertoire processed s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Lecture du dataset
    house_price = pd.read_csv(input_path)
    
    # Sélection des colonnes numériques et suppression des NA
    df_num = house_price.select_dtypes(include=['int64', 'float64']).dropna()
    
    # Sauvegarde du dataset nettoyé
    processed_path = os.path.join(output_dir, "train_processed.csv")
    df_num.to_csv(processed_path, index=False)
    print(f"Dataset préparé et sauvegardé dans : {processed_path}")
    
    return df_num

if __name__ == "__main__":
    prepare_data()
