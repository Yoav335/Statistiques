import pandas as pd
import numpy as np  # Import nécessaire pour np.number

# Chargement des données
df = pd.read_csv("data_stats.csv", delimiter=";")

# Supprimer les lignes où "Taux d’insertion" est NaN
df_cleaned = df.dropna(subset=["Taux d’insertion"])

# Supprimer les lignes où "Taux d’insertion" contient des valeurs non numériques
df_cleaned = df_cleaned[pd.to_numeric(df_cleaned["Taux d’insertion"], errors="coerce").notna()]

# Identifier les colonnes avant et y compris "Taux d’insertion" + inclure "Taux de chômage national"
if "Taux de chômage national" in df.columns:
    cols_to_keep = df.columns[: df.columns.get_loc("Taux de chômage national", "Nombre de réponses") + 1]
else:
    cols_to_keep = df.columns[: df.columns.get_loc("Taux d’insertion") + 1]  # En cas d'absence

# Garder uniquement les colonnes nécessaires
df_cleaned = df_cleaned.loc[:, cols_to_keep]

# Supprimer les colonnes avec plus de 50% de valeurs manquantes
missing_threshold = 0.5
df_cleaned = df_cleaned.loc[:, df_cleaned.isnull().mean() < missing_threshold]

# Remplacer les valeurs manquantes restantes par la moyenne pour les colonnes numériques
for col in df_cleaned.select_dtypes(include=[np.number]):
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

# Supprimer les lignes restantes qui contiennent encore des valeurs manquantes
df_cleaned = df_cleaned.dropna(axis=0)

# Vérification post-traitement
print("\nValeurs manquantes après traitement :")
print(df_cleaned.isnull().sum())

print(f"\nNombre de lignes et colonnes après nettoyage : {df_cleaned.shape}")

# Supprimer la colonne "Nombre de réponses" si elle existe
if "Nombre de réponses" in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=["Nombre de réponses"])

# Ajouter une nouvelle colonne basée sur "Code du domaine"
# Liste des codes de domaine ciblés
codes_cibles_domaine = ["STS"]

# Créer une nouvelle colonne après "Code du domaine"
df_cleaned.insert(
    loc=df_cleaned.columns.get_loc("Code du domaine") + 1,  # Insérer après "Code du domaine"
    column="Domaine ciblé",  # Nom de la nouvelle colonne
    value=df_cleaned["Code du domaine"].apply(lambda x: 1 if x in codes_cibles_domaine else 0)  # Appliquer la transformation
)

# Transformation de la colonne "Année" en format datetime
if "Année" in df_cleaned.columns:
    try:
        # Suppression des espaces éventuels et conversion en entier
        df_cleaned["Année"] = df_cleaned["Année"].astype(str).str.strip().str.replace(',', '').astype(int)
        
        # Transformation en format datetime (1er janvier de chaque année)
        df_cleaned["Année"] = pd.to_datetime(df_cleaned["Année"].astype(str) + "-01-01")

        print("\nLa colonne 'Année' a été convertie en format datetime.")
    except Exception as e:
        print(f"\nErreur lors de la conversion de la colonne 'Année' : {e}")
else:
    print("\nLa colonne 'Année' n'existe pas dans le jeu de données.")

# Sauvegarder le DataFrame nettoyé dans un fichier CSV
df_cleaned.to_csv("cleaned_data.csv", index=False)
