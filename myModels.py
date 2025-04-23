# -*- coding: utf-8 -*-
"""
myModels.ipynb
Application Streamlit pour entraîner des modèles sur un jeu de données CSV ou prédire des malwares à partir d'un fichier exécutable.
"""

# Installation des bibliothèques nécessaires (uniquement pour Colab ou environnements similaires)
!pip install pandas joblib scikit-learn streamlit optuna pefile

# Importation des bibliothèques
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import optuna
import pefile
import io
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Fonction pour extraire les caractéristiques d'un fichier exécutable
def extraire_caracteristiques_pe(contenu_fichier):
    try:
        # Charger le fichier PE à partir des octets
        pe = pefile.PE(data=contenu_fichier)
        
        # Définir les caractéristiques à extraire (basées sur l'analyse de malwares)
        caracteristiques = {
            'Machine': pe.FILE_HEADER.Machine,
            'TailleEnTeteOptionnel': pe.FILE_HEADER.SizeOfOptionalHeader,
            'Caracteristiques': pe.FILE_HEADER.Characteristics,
            'VersionLieurMajeure': pe.OPTIONAL_HEADER.MajorLinkerVersion,
            'VersionLieurMineure': pe.OPTIONAL_HEADER.MinorLinkerVersion,
            'TailleCode': pe.OPTIONAL_HEADER.SizeOfCode,
            'TailleDonneesInitialisees': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'TailleDonneesNonInitialisees': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            'AdressePointEntree': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'BaseCode': pe.OPTIONAL_HEADER.BaseOfCode,
            'BaseImage': pe.OPTIONAL_HEADER.ImageBase,
            'AlignementSection': pe.OPTIONAL_HEADER.SectionAlignment,
            'AlignementFichier': pe.OPTIONAL_HEADER.FileAlignment,
            'VersionOSMajeure': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            'VersionOSMineure': pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            'VersionImageMajeure': pe.OPTIONAL_HEADER.MajorImageVersion,
            'VersionImageMineure': pe.OPTIONAL_HEADER.MinorImageVersion,
            'VersionSousSystemeMajeure': pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            'VersionSousSystemeMineure': pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            'TailleImage': pe.OPTIONAL_HEADER.SizeOfImage,
            'TailleEnTetes': pe.OPTIONAL_HEADER.SizeOfHeaders,
            'SommeControle': pe.OPTIONAL_HEADER.CheckSum,
            'SousSysteme': pe.OPTIONAL_HEADER.Subsystem,
            'CaracteristiquesDLL': pe.OPTIONAL_HEADER.DllCharacteristics,
            'NbSections': len(pe.sections),
        }
        
        # Ajouter des caractéristiques spécifiques aux sections (taille, entropie)
        for i, section in enumerate(pe.sections[:3]):  # Limiter aux 3 premières sections
            caracteristiques[f'Section{i+1}_Taille'] = section.SizeOfRawData
            caracteristiques[f'Section{i+1}_Entropie'] = section.get_entropy()
        
        # Convertir en DataFrame
        return pd.DataFrame([caracteristiques])
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des caractéristiques : {str(e)}")
        return None

# Titre de l'application Streamlit
st.title("Application de Classification de Malwares")

# Widget pour télécharger un fichier
uploaded_file = st.file_uploader("Téléchargez un fichier CSV ou un exécutable (.exe, .dll)", type=["csv", "exe", "dll"])

if uploaded_file is not None:
    extension_fichier = uploaded_file.name.split('.')[-1].lower()
    
    if extension_fichier == 'csv':
        # --- Téléchargement CSV : Entraînement des modèles ---
        st.subheader("Entraînement des Modèles sur un Jeu de Données CSV")
        try:
            donnees = pd.read_csv(uploaded_file)
            st.write(f"Jeu de données chargé avec succès ! Dimensions : {donnees.shape}")

            # Vérifier la présence de la colonne 'legitimate'
            if "legitimate" not in donnees.columns:
                st.error("Le jeu de données doit contenir une colonne 'legitimate' pour la variable cible.")
            else:
                # Préparer les caractéristiques et la cible
                X = donnees.drop(columns=["legitimate"])
                y = donnees["legitimate"]

                # Diviser les données
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                st.write(f"Dimensions de l'ensemble d'entraînement : {X_train.shape}")
                st.write(f"Dimensions de l'ensemble de test : {X_test.shape}")

                # Standardiser les caractéristiques
                normaliseur = StandardScaler()
                X_train_normalise = normaliseur.fit_transform(X_train)
                X_test_normalise = normaliseur.transform(X_test)
                joblib.dump(normaliseur, 'normaliseur.pkl')  # Sauvegarder le normaliseur

                # --- Arbre de Décision ---
                st.subheader("Arbre de Décision")
                modele_dt = DecisionTreeClassifier()
                modele_dt.fit(X_train, y_train)
                predictions_dt = modele_dt.predict(X_test)
                precision_dt = accuracy_score(y_test, predictions_dt)
                st.write(f"Précision : {precision_dt:.2f}")
                joblib.dump(modele_dt, 'modele_dt.pkl')

                # --- SVM ---
                st.subheader("Machine à Vecteurs de Support (SVM)")
                modele_svm = SVC()
                modele_svm.fit(X_train_normalise, y_train)
                predictions_svm = modele_svm.predict(X_test_normalise)
                precision_svm = accuracy_score(y_test, predictions_svm)
                st.write(f"Précision : {precision_svm:.2f}")
                joblib.dump(modele_svm, 'modele_svm.pkl')

                # --- K-Voisins ---
                st.subheader("K-Voisins")
                modele_knn = KNeighborsClassifier(n_neighbors=5)
                modele_knn.fit(X_train_normalise, y_train)
                predictions_knn = modele_knn.predict(X_test_normalise)
                precision_knn = accuracy_score(y_test, predictions_knn)
                st.write(f"Précision : {precision_knn:.2f}")
                joblib.dump(modele_knn, 'modele_knn.pkl')

                # --- Forêt Aléatoire ---
                st.subheader("Forêt Aléatoire")
                modele_rf = RandomForestClassifier(random_state=42)
                modele_rf.fit(X_train, y_train)
                predictions_rf = modele_rf.predict(X_test)
                precision_rf = accuracy_score(y_test, predictions_rf)
                st.write(f"Précision : {precision_rf:.2f}")
                joblib.dump(modele_rf, 'modele_rf.pkl')

                # --- Optimisation des hyperparamètres avec Optuna (Arbre de Décision) ---
                st.subheader("Optimisation Optuna (Arbre de Décision)")
                def objectif(trial):
                    profondeur_max = trial.suggest_int('profondeur_max', 10, 50)
                    division_min = trial.suggest_int('division_min', 2, 20)
                    feuille_min = trial.suggest_int('feuille_min', 1, 10)
                    critere = trial.suggest_categorical('critere', ['gini', 'entropy'])
                    modele = DecisionTreeClassifier(
                        max_depth=profondeur_max,
                        min_samples_split=division_min,
                        min_samples_leaf=feuille_min,
                        criterion=critere
                    )
                    return cross_val_score(modele, X_train, y_train, cv=3, scoring='accuracy').mean()

                etude = optuna.create_study(direction='maximize')
                etude.optimize(objectif, n_trials=20)  # Réduction des essais pour accélérer
                st.write(f"Meilleurs hyperparamètres : {etude.best_params}")
                st.write(f"Meilleur score : {etude.best_value:.2f}")

                # --- Optimisation Forêt Aléatoire ---
                st.subheader("Optimisation Forêt Aléatoire")
                param_dist = {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
                recherche_aleatoire = RandomizedSearchCV(
                    estimator=RandomForestClassifier(random_state=42),
                    param_distributions=param_dist,
                    n_iter=10,
                    cv=3,
                    random_state=42,
                    n_jobs=-1
                )
                recherche_aleatoire.fit(X_train, y_train)
                meilleur_modele = recherche_aleatoire.best_estimator_
                predictions_optimisees = meilleur_modele.predict(X_test)
                st.write(f"Précision optimisée : {accuracy_score(y_test, predictions_optimisees):.2f}")

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier CSV : {str(e)}")

    elif extension_fichier in ['exe', 'dll']:
        # --- Téléchargement Exécutable : Prédiction de Malware ---
        st.subheader("Prédiction de Malware à partir d'un Exécutable")
        try:
            # Lire le contenu du fichier
            contenu_fichier = uploaded_file.read()
            
            # Extraire les caractéristiques
            df_caracteristiques = extraire_caracteristiques_pe(contenu_fichier)
            if df_caracteristiques is None:
                raise ValueError("Échec de l'extraction des caractéristiques.")

            # Charger les modèles pré-entraînés et le normaliseur
            try:
                modele_dt = joblib.load('modele_dt.pkl')
                modele_svm = joblib.load('modele_svm.pkl')
                modele_knn = joblib.load('modele_knn.pkl')
                modele_rf = joblib.load('modele_rf.pkl')
                normaliseur = joblib.load('normaliseur.pkl')
            except FileNotFoundError:
                st.error("Modèles pré-entraînés ou normaliseur introuvables. Veuillez d'abord entraîner les modèles avec un fichier CSV.")
                st.stop()

            # Aligner les caractéristiques avec les données d'entraînement
            try:
                colonnes_attendues = modele_dt.feature_names_in_  # Suppose que les modèles stockent les noms des caractéristiques
                colonnes_manquantes = [col for col in colonnes_attendues if col not in df_caracteristiques.columns]
                colonnes_excedentes = [col for col in df_caracteristiques.columns if col not in colonnes_attendues]

                # Gérer les colonnes manquantes ou excédentaires
                for col in colonnes_manquantes:
                    df_caracteristiques[col] = 0  # Remplir les colonnes manquantes avec des zéros
                df_caracteristiques = df_caracteristiques[colonnes_attendues]  # Réorganiser selon les colonnes attendues
            except AttributeError:
                st.error("Les modèles ne contiennent pas les noms des caractéristiques. Veuillez vérifier la compatibilité des caractéristiques.")
                st.stop()

            # Normaliser les caractéristiques pour SVM et KNN
            caracteristiques_normalisees = normaliseur.transform(df_caracteristiques)

            # Effectuer les prédictions
            pred_dt = modele_dt.predict(df_caracteristiques)[0]
            pred_svm = modele_svm.predict(caracteristiques_normalisees)[0]
            pred_knn = modele_knn.predict(caracteristiques_normalisees)[0]
            pred_rf = modele_rf.predict(df_caracteristiques)[0]

            # Afficher les résultats
            st.write("Résultats des Prédictions :")
            st.write(f"Arbre de Décision : {'Malware' if pred_dt == 0 else 'Légitime'}")
            st.write(f"SVM : {'Malware' if pred_svm == 0 else 'Légitime'}")
            st.write(f"K-Voisins : {'Malware' if pred_knn == 0 else 'Légitime'}")
            st.write(f"Forêt Aléatoire : {'Malware' if pred_rf == 0 else 'Légitime'}")

        except Exception as e:
            st.error(f"Erreur lors du traitement de l'exécutable : {str(e)}")

else:
    st.info("Veuillez télécharger un fichier CSV pour entraîner les modèles ou un fichier exécutable pour prédire un malware.")