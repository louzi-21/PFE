import streamlit as st
import pandas as pd
import os
import sys
from sqlalchemy import create_engine

# Pour importer clean.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clean import clean_fournisseur

# Connexion MySQL
engine = create_engine('mysql+mysqlconnector://root:louzisql@localhost:3306/pfe')

# Chargement des données nettoyées
df = clean_fournisseur(engine).reset_index(drop=True)

# Fichier d'annotations
ANNOTATION_FILE = "annotations_fournisseur.csv"

# Chargement des annotations existantes
if os.path.exists(ANNOTATION_FILE):
    annotations = pd.read_csv(ANNOTATION_FILE)
else:
    annotations = pd.DataFrame(columns=df.columns.tolist() + ['annotation'])

# Index de démarrage
st.title("📝 Interface d’annotation - Fournisseurs")
st.write("Indiquez si chaque ligne représente une **anomalie** ou un cas **normal**.")

# Trouver les lignes non encore annotées
annotated_ids = annotations['IDFournisseur'].tolist()
df_to_annotate = df[~df['IDFournisseur'].isin(annotated_ids)]

if df_to_annotate.empty:
    st.success("🎉 Toutes les lignes ont été annotées !")
else:
    row = df_to_annotate.iloc[0]
    with st.form(key="annotation_form"):
        st.write("### Informations du fournisseur à annoter :")
        st.dataframe(pd.DataFrame([row]))

        label = st.radio("Annotation :", ["Normale", "Anomalie"])
        submit = st.form_submit_button("Enregistrer")

        if submit:
            row_dict = row.to_dict()
            row_dict["annotation"] = 0 if label == "Normale" else 1
            annotations = pd.concat([annotations, pd.DataFrame([row_dict])], ignore_index=True)
            annotations.to_csv(ANNOTATION_FILE, index=False)
            st.success(f"Ligne annotée comme : {label}")
            st.experimental_rerun()
