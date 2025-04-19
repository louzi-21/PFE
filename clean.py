import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import os
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# --- Paramètres de connexion (modifiez ces valeurs selon votre configuration) ---
username = 'root'
password = 'louzisql'
host = 'localhost'
port = '3306'
database = 'pfe'

# Création de la connexion à la base de données MySQL
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}')

###############################################################################
# Fonction utilitaire pour corriger toutes les colonnes texte d'un DataFrame
###############################################################################
def fix_nan_in_all_text_cols(df):
    """
    Pour chaque colonne de type 'object' (texte), on :
      1) Convertit la colonne en str et strip() les espaces.
      2) Remplace les chaînes vides ("") par np.nan.
      3) Remplace "nan" (ignorer la casse) par np.nan.
    """
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        # 1. Convertir en str et enlever les espaces autour
        df[col] = df[col].astype(str).str.strip()
        # 2. Remplacer la chaîne vide par NaN
        df[col].replace("", np.nan, inplace=True)
        # 3. Remplacer "nan" (ignorer la casse) par NaN
        df[col].replace(r'(?i)^nan$', np.nan, regex=True, inplace=True)
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "ar_sfamille" ---
###############################################################################
def clean_ar_sfamille(engine):
    df = pd.read_sql("SELECT * FROM ar_sfamille", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    # Suppression des colonnes non nécessaires
    df.drop(['IDChaineMontage', 'IDCategorieOFpardefaut'], axis=1, inplace=True)
    
    # Conversion en numérique pour les colonnes critiques
    numeric_cols = ['IDArSousFamille', 'Etat', 'IDArFamille']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Suppression des lignes où ces colonnes sont NULL
    df.dropna(subset=numeric_cols, inplace=True)
    
    # Règle métier : Etat doit être 0 ou 1
    df = df[df['Etat'].isin([0, 1])]
    
    # Suppression des doublons
    df.drop_duplicates(inplace=True)
    
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "arfamille" ---
###############################################################################
def clean_arfamille(engine):
    df = pd.read_sql("SELECT * FROM arfamille", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    # Suppression des colonnes non nécessaires
    df.drop(['IDChaineMontage', 'QtePPP', 'Type', 'CodeDouane'], axis=1, inplace=True)
    
    # Conversion en numérique pour les colonnes critiques
    numeric_cols = ['IDArFamille', 'Etat', 'SaisonObligatoire']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=numeric_cols, inplace=True)
    
    # Règle métier : Etat doit être 0 ou 1, SaisonObligatoire doit être 0 ou 1
    df = df[df['Etat'].isin([0, 1])]
    df = df[df['SaisonObligatoire'].isin([0, 1])]
    
    df.drop_duplicates(inplace=True)
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "article" ---
###############################################################################
def clean_article(engine):
    df = pd.read_sql("SELECT * FROM article", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    # Liste combinée des colonnes à supprimer
    drop_columns = [
        "IDGamme", "IDClient", "TempsClient", "IdProcess", "prixMP", "Valeur", 
        "Cadence", "IdArticleBase", "SemiFini", "ValeurTissu", "ValeurFourniture", 
        "ValeurMP", "TypeTarif", "IdMeilleurOF", "BaseStylisme", "IDTypeMatiereBase", 
        "IDVarianteModele", "IDGenre", "IDBroderie", "IDSerigraphie", "IDGarniture", 
        "IDTypeAccessoire", "IDTransfert", "IDCouleurGarniture", "IDCouleurBroderie", 
        "IDCouleurSerigraphie", "PrixEmballage", "StockMin", "StockAlerte", "ValeurMPEuro", 
        "ValeurMPAutre", "ValeurMPTunisie", "ValeurMPEuromed", "AQL", "AQLMineur", 
        "IDNiveauControle", "AQLCritique", "IDCategorie", "IDCategoriereclamation", 
        "IDCartouche", "IDArticleParent", "isParent", "QteFils", "Dimensions", 
        "TempsAtelier", "TempsFinitions", "IDTypeMatelassage", "IDMP", "IsMP", 
        "IDDecorArticle", "IsSemiFini", "TempsUnitaire", "TauxSondageQlte", "IDNorme", 
        "DDV", "FraisTransport", "AutresFrais", "IDArticleEtqEntretien", "Ecologique", 
        "TauxDefectueux", "Publier", "Ordre", "TauxCommissionCA", "CODE_OLD", "PrixEtude",
        "CodeDouane", "Observations", "NomenclatureValidePar", "NbrPiecesColis", "NbrColisPalette",
        "PoidsEmballage", "IDcomplexite", "IDAr_Theme", "Emballage", "Boutonnage", 
        "SupportArt", "ReseauArt", "ReferenceFssr", "IDFibreComposition", "PrixOutlet", "IDPlanComptable"
    ]
    df.drop(columns=drop_columns, inplace=True)
    
    # Conversion en numérique pour quelques colonnes critiques
    numeric_cols = ["IDArticle", "Etat", "TauxTVA", "NumInterne", "IDSaison"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Règle métier : Etat doit être 0 ou 1
    df = df[df["Etat"].isin([0, 1])]
    
    # Conversion des colonnes de type date
    date_cols = ["SaisiLe", "ModifieLe"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Suppression des lignes essentielles manquantes
    essential_cols = ["IDArticle", "Code"]
    df.dropna(subset=essential_cols, inplace=True)
    
    # Suppression des doublons éventuels
    df.drop_duplicates(inplace=True)
    
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "codebarre" ---
###############################################################################
def clean_codebarre(engine):
    df = pd.read_sql("SELECT * FROM codebarre", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    # Supprimer les colonnes non nécessaires
    df.drop(columns=["Indice", "IDSerieArticle","NumInterne"], inplace=True)
    
    # Conversion en numérique pour les colonnes critiques
    numeric_cols = ['IDCodeBarre', 'IdEntite', 'IdTaille', 'IDAr_Couleur', 'Prix', 'isSynchronized', 'isSynchronizedWeb']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Règle métier : isSynchronized et isSynchronizedWeb doivent être 0 ou 1
    for col in ['isSynchronized', 'isSynchronizedWeb']:
        df = df[df[col].isin([0, 1])]
    
    # Règle métier additionnelle : Prix doit être > 0
    df = df[df["Prix"] > 0]
    
    # Traitement des valeurs manquantes dans les colonnes essentielles
    df.dropna(subset=['IDCodeBarre', 'CodeBarre'], inplace=True)
    
    # Suppression des doublons éventuels
    df.drop_duplicates(inplace=True)
    
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "fournisseur" ---
###############################################################################
def clean_fournisseur(engine):
    df = pd.read_sql("SELECT * FROM fournisseur", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    drop_columns = [
        "isFournisseur", "Note", "Type", "FournitMP", "FournitMB", "NumInterne",
        "TauxRetenueSource", "ExonerationRS", "IsPDR", "Timbre", "ToleranceMAxAccepte",
        "IDBanque", "AdresseBanque", "VilleBanque", "NumCompte", "CodeSwift", "IBAN",
        "NonAssujettiTVA", "DelaisLivraison", "Reference", "IDFournisseurParent", "Difference",
        "AppliqueFodec", "Login_FRS", "IDPlanComptable", "DateExonerationRS", "Echeance",
        "IDConditionReglement"
    ]
    df.drop(columns=drop_columns, inplace=True)
    
    # Conversion en numérique pour les colonnes critiques restantes
    numeric_columns = ["IDFournisseur", "Chiffre", "Reglements", "Solde", "IDDevise", "IDCategorie", "IDPays", "Etat", "IDCGAFournisseur", "IsMP", "IsPF"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Règle métier : Etat doit être 0 ou 1
    df = df[df["Etat"].isin([0, 1])]
    # Règle métier : FournitPF doit être 0 ou 1 (si la colonne est conservée)
    
    # Suppression des lignes essentielles manquantes
    essential_cols = ["IDFournisseur", "Fournisseur", "Code"]
    df.dropna(subset=essential_cols, inplace=True)
    
    # Suppression des doublons éventuels
    df.drop_duplicates(inplace=True)
    
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "saison" ---
###############################################################################
def clean_saison(engine):
    df = pd.read_sql("SELECT * FROM saison", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    # Supprimer les colonnes DateDebut et DateFin
    df.drop(columns=["DateDebut", "DateFin"], inplace=True)
    
    # Conversion en numérique pour les colonnes critiques
    numeric_cols = ['IDSaison', 'Etat', 'IDTypeSaison']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Règle métier : Etat doit être 0 ou 1
    df = df[df["Etat"].isin([0, 1])]
    
    # Suppression des lignes essentielles manquantes
    df.dropna(subset=['IDSaison', 'Saison', 'Code'], inplace=True)
    
    # Suppression des doublons
    df.drop_duplicates(inplace=True)
    
    return df

###############################################################################
# --- Fonction de nettoyage pour la table "tailles" ---
###############################################################################
def clean_tailles(engine):
    df = pd.read_sql("SELECT * FROM tailles", con=engine)
    
    # Corriger toutes les colonnes texte
    df = fix_nan_in_all_text_cols(df)
    
    # Supprimer les colonnes non nécessaires
    drop_columns = ["LibTailleAR", "LibTailleAutre", "LibTailleGER", "LibTailleUSA", "LibTailleSP", "LibTailleGRK"]
    df.drop(columns=drop_columns, inplace=True)
    
    # Conversion en numérique pour les colonnes critiques
    numeric_cols = ['IDGrille', 'IdTaille', 'Ordre', 'isMilieu']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Suppression des lignes essentielles manquantes
    df.dropna(subset=['LibTaille', 'IdTaille'], inplace=True)
    
    # Règle métier : isMilieu doit être 0 ou 1
    df = df[df["isMilieu"].isin([0, 1])]
    
    # Validation pour Ordre (ici, on vérifie qu'il est dans la plage 0-255)
    df = df[(df["Ordre"] >= 0) & (df["Ordre"] <= 255)]
    
    # Suppression des doublons
    df.drop_duplicates(inplace=True)
    
    return df

###############################################################################
# --- Exécution du nettoyage et génération des rapports pour toutes les tables
###############################################################################
if __name__ == '__main__':
    print("=== Début du nettoyage des tables ===\n")
    
    df_ar_sfamille_clean = clean_ar_sfamille(engine)
    df_arfamille_clean = clean_arfamille(engine)
    df_article_clean = clean_article(engine)
    df_codebarre_clean = clean_codebarre(engine)
    df_fournisseur_clean = clean_fournisseur(engine)
    df_saison_clean = clean_saison(engine)
    df_tailles_clean = clean_tailles(engine)
    
    # Génération des rapports pour chaque table
    ProfileReport(df_ar_sfamille_clean, title="Rapport Exploratoire - ar_sfamille", explorative=True)\
        .to_file("rapport_ar_sfamille.html")
    ProfileReport(df_arfamille_clean, title="Rapport Exploratoire - arfamille", explorative=True)\
        .to_file("rapport_arfamille.html")
    ProfileReport(df_article_clean, title="Rapport Exploratoire - article", explorative=True)\
        .to_file("rapport_article.html")
    ProfileReport(df_codebarre_clean, title="Rapport Exploratoire - codebarre", explorative=True)\
        .to_file("rapport_codebarre.html")
    ProfileReport(df_fournisseur_clean, title="Rapport Exploratoire - fournisseur", explorative=True)\
        .to_file("rapport_fournisseur.html")
    ProfileReport(df_saison_clean, title="Rapport Exploratoire - saison", explorative=True)\
        .to_file("rapport_saison.html")
    ProfileReport(df_tailles_clean, title="Rapport Exploratoire - tailles", explorative=True)\
        .to_file("rapport_tailles.html")
    
    print("\nLes rapports HTML ont été générés dans le répertoire :", os.getcwd())
    print("\n=== Fin du nettoyage et de l'analyse exploratoire ===")
