import mysql.connector
import pandas as pd
from ydata_profiling import ProfileReport

# Configuration de la connexion MySQL
config = {
    'host': 'localhost',           # Adresse du serveur MySQL
    'user': 'root',                # Nom d'utilisateur MySQL (modifiez si nécessaire)
    'password': 'louzisql',                # Mot de passe MySQL (modifiez si nécessaire)
    'database': 'pfe'  # Nom de votre base de données (modifiez si nécessaire)
}

# Connexion à la base de données
try:
    conn = mysql.connector.connect(**config)
    print("Connexion réussie à la base de données.")
except mysql.connector.Error as err:
    print(f"Erreur lors de la connexion : {err}")
    exit(1)

# Liste des tables connues dans votre projet
tables = [
    'ar_sfamille',
    'arfamille',
    'article',
    'codebarre',
    'fournisseur',
    'grille',
    'saison',
    'tailles'
]

# Génération des rapports exploratoires pour chaque table
for table_name in tables:
    try:
        print(f"Traitement de la table '{table_name}'...")
        # Lecture des données de la table dans un DataFrame
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        # Génération du rapport exploratoire
        report = ProfileReport(df, title=f"Rapport Exploratoire - {table_name}", explorative=True)
        output_file = f"rapport_{table_name}.html"
        report.to_file(output_file)
        print(f"Rapport généré pour '{table_name}' : {output_file}\n")
    except Exception as e:
        print(f"Erreur lors du traitement de la table '{table_name}' : {e}\n")

# Fermeture de la connexion
conn.close()
print("Connexion fermée.")
