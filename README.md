# 🕵️‍♂️ Détection des Transactions Frauduleuses – Attijari Bank

## 📌 Description du projet
Ce projet vise à détecter automatiquement les transactions bancaires suspectes en utilisant des techniques de Machine Learning non supervisées : SVM, AutoEncoder, et Isolation Forest.  
Il inclut également une application web interactive développée avec **Streamlit** ainsi qu’un **dashboard Power BI** pour la visualisation des données.

## 🧠 Modèles utilisés
- 🔹 One-Class SVM
- 🔹 AutoEncoder (Keras)
- 🔹 Isolation Forest

## 🗂️ Structure du projet

Detection_transc_fraudes/
│
├── code_app/ # Application Streamlit
│ └── app.py
│ └── users.xlsx
│
├── data/ # Fichiers simulés (.csv)
│ └── transactions_fraudes.csv
│ └── transactions_atijari_bank.csv
│ └── transactions_analysees.csv
│
├── models/ # Modèles ML en notebooks
│ └── One_classe_svm.ipynb
│ └── Transac_frauduleuses.ipynb
│ └── transac_fradu_autoencoder.ipynb
│
├── visualisation/
│ └── propre_projet.pbix # Power BI
│
└── README.md
