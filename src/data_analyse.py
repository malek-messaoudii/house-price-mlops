# src/analyze_data.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_and_clean(input_path="data/train.csv", output_path="data/train.csv", corr_threshold=0.1):
    """
    Analyse la corrélation des features avec la cible SalePrice,
    supprime les colonnes peu utiles et sauvegarde le dataset nettoyé.
    
    Args:
        input_path (str): chemin du dataset original
        output_path (str): chemin pour sauvegarder le dataset nettoyé
        corr_threshold (float): seuil de corrélation minimum pour garder une colonne
    """
    
    # --- Lecture du dataset ---
    df = pd.read_csv(input_path)
    print(f"Dataset original : {df.shape} lignes x colonnes")
    
    # --- Sélection des colonnes numériques ---
    df_num = df.select_dtypes(include=['int64', 'float64']).dropna()
    
    # --- Calcul de la corrélation avec la cible SalePrice ---
    corr_matrix = df_num.corr()
    
    # --- Visualisation des corrélations avec SalePrice ---
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix[['SalePrice']].sort_values(by='SalePrice', ascending=False),
                annot=True, cmap='coolwarm')
    plt.title("Corrélation des features avec SalePrice")
    plt.show()
    
    # --- Identifier les colonnes peu corrélées ---
    low_corr_cols = corr_matrix['SalePrice'][corr_matrix['SalePrice'].abs() < corr_threshold].index.tolist()
    print(f"Colonnes peu corrélées (corr < {corr_threshold}) :", low_corr_cols)
    
    # --- Supprimer ces colonnes ---
    df_clean = df_num.drop(columns=low_corr_cols)
    print(f"Dataset nettoyé : {df_clean.shape} lignes x colonnes")
    
    # --- Créer le dossier de sortie si nécessaire ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --- Sauvegarder le dataset nettoyé ---
    df_clean.to_csv(output_path, index=False)
    print(f"Dataset nettoyé sauvegardé dans : {output_path}")
    
    return df_clean

# --- Appel de la fonction si le script est exécuté directement ---
if __name__ == "__main__":
    analyze_and_clean()
