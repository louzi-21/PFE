import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import plotly.express as px

# Importer depuis clean.py
from clean import clean_codebarre

# Connexion à MySQL
engine = create_engine('mysql+mysqlconnector://root:louzisql@localhost:3306/pfe')
df = clean_codebarre(engine)

# Standardisation
X_numeric = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# --- Isolation Forest ---
model_if = IsolationForest(contamination=0.01, random_state=42)
df["iforest"] = (model_if.fit_predict(X_scaled) == -1).astype(int)

# --- Autoencodeur ---
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
output_layer = Dense(input_dim)(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, shuffle=True, verbose=0)

reconstructions = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)
df["autoenc"] = (mse > threshold).astype(int)

# Comparaison
df["both"] = ((df["iforest"] == 1) & (df["autoenc"] == 1)).astype(int)
df["only_iforest"] = ((df["iforest"] == 1) & (df["autoenc"] == 0)).astype(int)
df["only_autoenc"] = ((df["iforest"] == 0) & (df["autoenc"] == 1)).astype(int)

# PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Libellés
df["type_anomalie"] = "Normale"
df.loc[df["only_iforest"] == 1, "type_anomalie"] = "Isolation Forest seulement"
df.loc[df["only_autoenc"] == 1, "type_anomalie"] = "Autoencodeur seulement"
df.loc[df["both"] == 1, "type_anomalie"] = "Anomalie commune"

# --- Résumé HTML ---
html_summary = f"""
<h2>Résumé Comparatif</h2>
<ul>
  <li><strong>Total lignes :</strong> {len(df)}</li>
  <li><strong>Anomalies Isolation Forest :</strong> {df['iforest'].sum()}</li>
  <li><strong>Anomalies Autoencodeur :</strong> {df['autoenc'].sum()}</li>
  <li><strong>Anomalies communes :</strong> {df['both'].sum()}</li>
  <li><strong>Seulement Isolation Forest :</strong> {df['only_iforest'].sum()}</li>
  <li><strong>Seulement Autoencodeur :</strong> {df['only_autoenc'].sum()}</li>
</ul>
"""

# --- Exemples HTML (exclusifs) ---
def generate_preview_html(df_part, title, max_rows=10):
    html_table = df_part.head(max_rows).to_html(index=False, classes='preview-table', border=1)
    return f"<h3>{title}</h3>{html_table}<br>"

# Sélection de sous-groupes exclusifs
df_communes = df[df["both"] == 1].copy()
df_only_if = df[df["only_iforest"] == 1].copy()
df_only_ae = df[df["only_autoenc"] == 1].copy()

# Regrouper IF et AE (avec communes mais en enlevant les doublons)
df_iforest_total = pd.concat([df_only_if, df_communes]).drop_duplicates().head(10)
df_autoenc_total = pd.concat([df_only_ae, df_communes]).drop_duplicates().head(10)

html_previews = ""
html_previews += generate_preview_html(df_iforest_total, "🔹 Anomalies Isolation Forest (total)")
html_previews += generate_preview_html(df_autoenc_total, "🔹 Anomalies Autoencodeur (total)")
html_previews += generate_preview_html(df_communes, "✅ Anomalies communes (IF + AE)")
html_previews += generate_preview_html(df_only_if, "🔵 Seulement Isolation Forest")
html_previews += generate_preview_html(df_only_ae, "🟢 Seulement Autoencodeur")

# --- Graphique interactif ---
fig = px.scatter(
    df, x="PCA1", y="PCA2", color="type_anomalie",
    title="Détection d'anomalies - Isolation Forest vs Autoencodeur (PCA)",
    opacity=0.7, width=900, height=600
)

# --- Génération du rapport HTML ---
with open("rapport_comparatif_anomalies.html", "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'><style>h3 { margin-top: 30px; }</style></head><body>")
    f.write(html_summary)
    f.write(html_previews)
    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("</body></html>")

print("✅ Rapport interactif généré : rapport_comparatif_anomalies.html")
