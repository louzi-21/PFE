import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from clean import clean_codebarre

# Connexion base de données
engine = create_engine('mysql+mysqlconnector://root:louzisql@localhost:3306/pfe')
df = clean_codebarre(engine)

# Nettoyage
df.drop(columns=["Indice", "IDSerieArticle"], errors='ignore', inplace=True)
df.dropna(inplace=True)
df = df.select_dtypes(include=[np.number])
df.drop_duplicates(inplace=True)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
df["anomaly"] = model.fit_predict(X_scaled)
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

# Résumé
total = len(df)
anomalies = df["anomaly"].sum()
print(f"\n🔍 Isolation Forest : {anomalies} anomalies détectées sur {total} lignes\n")
print("📌 Aperçu des 10 premières anomalies :\n")
print(df[df["anomaly"] == 1].head(10).to_string(index=False))

# PCA pour visualisation
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df["PCA1"] = components[:, 0]
df["PCA2"] = components[:, 1]

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(df[df.anomaly == 0]["PCA1"], df[df.anomaly == 0]["PCA2"], 
            c='blue', label='Normaux', alpha=0.5)
plt.scatter(df[df.anomaly == 1]["PCA1"], df[df.anomaly == 1]["PCA2"], 
            c='red', label='Anomalies', alpha=0.7)
plt.legend()
plt.title("Détection d'anomalies - Isolation Forest (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()
