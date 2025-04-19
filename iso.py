import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Importer les fonctions de nettoyage et l'objet engine depuis clean.py
from clean import (
    clean_ar_sfamille,
    clean_arfamille,
    clean_article,
    clean_codebarre,
    clean_fournisseur,
    clean_saison,
    clean_tailles,
    engine
)

# Fonction générique de détection d'anomalies via Isolation Forest
def detect_anomalies(df, features, contamination=0.05):
    # Sélectionner les colonnes d'intérêt et supprimer les lignes avec des valeurs manquantes
    data = df[features].dropna().copy()
    # Instanciation d'Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    # Appliquer le modèle et ajouter la colonne "anomaly" (1 = normal, -1 = anomalie)
    data["anomaly"] = iso.fit_predict(data)
    return data, data[data["anomaly"] == -1]

# Fonction générique pour visualiser les anomalies
def plot_anomalies(data, x_feature, y_feature, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_feature], data[y_feature], c=data["anomaly"], cmap="coolwarm", alpha=0.6)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(title)
    plt.colorbar(label="1 = normal, -1 = anomalie")
    plt.show()

# Détection d'anomalies pour la table ar_sfamille
def anomaly_ar_sfamille():
    df = clean_ar_sfamille(engine)
    features = ["IDArSousFamille", "Etat", "IDArFamille"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table ar_sfamille - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : par défaut, tracer IDArSousFamille vs Etat
    plot_anomalies(data, "IDArSousFamille", "Etat", "Anomalies dans ar_sfamille")
    return data, anomalies

# Détection d'anomalies pour la table arfamille
def anomaly_arfamille():
    df = clean_arfamille(engine)
    features = ["IDArFamille", "Etat", "SaisonObligatoire"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table arfamille - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : tracer IDArFamille vs Etat
    plot_anomalies(data, "IDArFamille", "Etat", "Anomalies dans arfamille")
    return data, anomalies

# Détection d'anomalies pour la table article
def anomaly_article():
    df = clean_article(engine)
    features = ["IDArticle", "Etat", "TauxTVA", "NumInterne", "IDSaison"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table article - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : tracer TauxTVA vs Etat
    plot_anomalies(data, "TauxTVA", "Etat", "Anomalies dans article")
    return data, anomalies

# Détection d'anomalies pour la table codebarre
def anomaly_codebarre():
    df = clean_codebarre(engine)
    features = ["IDCodeBarre", "IdEntite", "IdTaille", "IDAr_Couleur", "Prix", "NumInterne", "isSynchronized", "isSynchronizedWeb"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table codebarre - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : tracer Prix vs IDCodeBarre (à adapter ultérieurement)
    plot_anomalies(data, "Prix", "IDCodeBarre", "Anomalies dans codebarre")
    return data, anomalies

# Détection d'anomalies pour la table fournisseur
def anomaly_fournisseur():
    df = clean_fournisseur(engine)
    features = ["IDFournisseur", "Chiffre", "Reglements", "Solde", "Etat", "FournitPF"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table fournisseur - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : tracer Chiffre vs Solde
    plot_anomalies(data, "Chiffre", "Solde", "Anomalies dans fournisseur")
    return data, anomalies

# Détection d'anomalies pour la table saison
def anomaly_saison():
    df = clean_saison(engine)
    features = ["IDSaison", "Etat", "IDTypeSaison"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table saison - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : tracer IDSaison vs Etat
    plot_anomalies(data, "IDSaison", "Etat", "Anomalies dans saison")
    return data, anomalies

# Détection d'anomalies pour la table tailles
def anomaly_tailles():
    df = clean_tailles(engine)
    features = ["IdTaille", "IDGrille", "Ordre", "isMilieu"]
    data, anomalies = detect_anomalies(df, features, contamination=0.05)
    print("Table tailles - Nombre d'anomalies détectées :", anomalies.shape[0])
    # Visualisation : tracer Ordre vs isMilieu
    plot_anomalies(data, "Ordre", "isMilieu", "Anomalies dans tailles")
    return data, anomalies

if __name__ == '__main__':
    print("=== Début de la détection d'anomalies par table ===\n")
    
    anomaly_ar_sfamille()
    anomaly_arfamille()
    anomaly_article()
    anomaly_codebarre()
    anomaly_fournisseur()
    anomaly_saison()
    anomaly_tailles()
    
    print("\n=== Fin de la détection d'anomalies ===")
