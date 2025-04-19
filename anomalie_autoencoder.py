
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from clean import clean_codebarre

# Connexion
engine = create_engine('mysql+mysqlconnector://root:louzisql@localhost:3306/pfe')
df = clean_codebarre(engine)

# Prétraitement
df.drop(columns=["Indice", "IDSerieArticle"], errors='ignore', inplace=True)
df.dropna(inplace=True)
df = df.select_dtypes(include=[np.number])
df.drop_duplicates(inplace=True)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Autoencodeur
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
output_layer = Dense(input_dim)(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, shuffle=True, verbose=0)

# Reconstruction
reconstructions = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)

df['anomaly'] = (mse > threshold).astype(int)

# Résumé
total = len(df)
anomalies = df["anomaly"].sum()
print(f"\n🤖 Autoencodeur : {anomalies} anomalies détectées sur {total} lignes\n")
print("📌 Aperçu des 10 premières anomalies :\n")
print(df[df["anomaly"] == 1].head(10).to_string(index=False))

# PCA
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
plt.title("Détection d'anomalies - Autoencodeur (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()
