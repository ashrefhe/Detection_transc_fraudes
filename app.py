
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os

USER_FILE = r"C:/Users/MSI/Desktop/Attijari_bank/propre_projet/users.xlsx"
USER_COLUMNS = ["username", "password", "nom", "prenom", "email", "sexe", "date_naissance", "service"]

if not os.path.exists(USER_FILE):
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    pd.DataFrame(columns=USER_COLUMNS).to_excel(USER_FILE, index=False)

if "page" not in st.session_state:
    st.session_state.page = "login"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def page_login():
    st.markdown("""
    <style>
    body, [data-testid="stAppViewContainer"] > .main {
        background-color: black;
        color: white;
    }
    label, .stTextInput > label, .stPasswordInput > label {
        color: white !important;
        font-weight: bold;
    }
    input {
        background-color: white !important;
        color: black !important;
    }
    .stButton > button {
        color: white !important;
        background-color: #444 !important;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.image("https://i.postimg.cc/NGWDd6zH/ecf453d4d1450cb06f275f65f22c8820.png", width=180)
    st.title("🔐 Connexion")
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Connexion")
        if submit:
            users = pd.read_excel(USER_FILE)
            match = users[(users["username"] == username) & (users["password"] == password)]
            if not match.empty:
                st.success(f"Bienvenue {match.iloc[0]['prenom']} {match.iloc[0]['nom']} 👋")
                st.session_state.authenticated = True
                st.session_state.page = "analyse"
                st.session_state.current_user = username
                st.rerun()
            else:
                st.error("Identifiants incorrects.")
    if st.button("Créer un compte"):
        st.session_state.page = "register"

def page_register():
    import datetime
    background()
    st.title("📝 Inscription")
    with st.form("form_register"):
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        email = st.text_input("Email")
        sexe = st.selectbox("Sexe", ["Homme", "Femme"])
        date_naissance = st.date_input("Date de naissance", min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
        service = st.text_input("Service")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        confirm = st.text_input("Confirmer mot de passe", type="password")
        submit = st.form_submit_button("Créer le compte")
        if submit:
            if password != confirm:
                st.warning("❗ Mots de passe différents.")
            elif username.strip() == "":
                st.warning("❗ Nom d'utilisateur obligatoire.")
            else:
                users = pd.read_excel(USER_FILE)
                if username in users['username'].values:
                    st.error("⚠️ Ce nom d'utilisateur existe déjà.")
                else:
                    new_user = pd.DataFrame([[username, password, nom, prenom, email, sexe, str(date_naissance), service]], columns=USER_COLUMNS)
                    users = pd.concat([users, new_user], ignore_index=True)
                    users.to_excel(USER_FILE, index=False)
                    st.success(f"✅ Compte créé : {prenom} {nom}")
                    st.session_state.page = "login"
    if st.button("⬅️ Retour"):
        st.session_state.page = "login"

def page_analyse():
    background()
    st.title("💳 Analyse de Transactions – Attijari Bank")
    users = pd.read_excel(USER_FILE)
    user = users[users["username"] == st.session_state.get("current_user")]
    if not user.empty:
        nom = user.iloc[0]["nom"]
        prenom = user.iloc[0]["prenom"]
        st.success(f"👋 Bienvenue {prenom} {nom}")
    uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=";")
        if 'heure' in df.columns:
            df['heure_num'] = df['heure'].str.split(':').apply(lambda x: int(x[0]) + int(x[1]) / 60)
        if 'type_transaction' in df.columns:
            df['type_tx_encoded'] = LabelEncoder().fit_transform(df['type_transaction'])
        if st.button("🔄 Réinitialiser les filtres"):
            st.rerun()
        st.subheader("🎛️ Filtres dynamiques")
        if 'type_transaction' in df.columns:
            txs = df['type_transaction'].dropna().unique().tolist()
            sel = st.multiselect("Type de transaction", txs, default=txs)
            df = df[df['type_transaction'].isin(sel)]
        if 'lieu' in df.columns:
            lieux = df['lieu'].dropna().unique().tolist()
            lieu_sel = st.multiselect("Lieu", lieux, default=lieux)
            df = df[df['lieu'].isin(lieu_sel)]
        if 'heure_num' in df.columns:
            h1, h2 = st.slider("⏰ Plage horaire", 0.0, 24.0, (0.0, 24.0), step=0.5, format="%0.1f h")
            if h1 < h2:
                df = df[(df['heure_num'] >= h1) & (df['heure_num'] <= h2)]
            else:
                df = df[(df['heure_num'] >= h1) | (df['heure_num'] <= h2)]
        all_features = ['montant', 'frequence', 'solde_avant', 'solde_après', 'heure_num', 'type_tx_encoded']
        features = [col for col in all_features if col in df.columns]
        if len(features) < 2:
            st.error("⚠️ Le fichier ne contient pas suffisamment de colonnes pour effectuer l'analyse.")
            return
        df = df.dropna(subset=features)
        X = df[features]
        if X.empty:
            st.warning("⚠️ Aucun enregistrement ne correspond aux filtres sélectionnés.")
            return
        X_scaled = MinMaxScaler().fit_transform(X)
        st.subheader("🤖 Choisir un modèle à exécuter")
        col1, col2, col3 = st.columns(3)
        if col1.button("Isolation Forest"):
            model = IsolationForest(contamination=0.05, random_state=42)
            df['anomaly'] = model.fit_predict(X)
            df['is_anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
            show_results(df, 'is_anomaly', "Isolation Forest")
        if col2.button("Autoencoder"):
            model = models.Sequential([
                layers.Input(shape=(X_scaled.shape[1],)),
                layers.Dense(4, activation='relu'),
                layers.Dense(2, activation='relu'),
                layers.Dense(4, activation='relu'),
                layers.Dense(X_scaled.shape[1], activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse')
            X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
            model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), verbose=0)
            preds = model.predict(X_scaled)
            mse = np.mean(np.square(X_scaled - preds), axis=1)
            seuil = np.percentile(mse, 95)
            df['is_anomaly'] = mse > seuil
            show_results(df, 'is_anomaly', "Autoencoder")
        if col3.button("One-Class SVM"):
            model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
            df['anomaly'] = model.fit_predict(X_scaled)
            df['is_anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
            show_results(df, 'is_anomaly', "One-Class SVM")
    st.markdown("""
    <style>
    .stButton>button {
        white-space: nowrap;
        padding: 0.5em 1.5em;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([8, 1, 2])
    with col3:
        if st.button("🔓 Se déconnecter"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.session_state.current_user = None
            st.rerun()

def show_results(df, col_anomaly, model_name):
    st.subheader("📟 Transactions détectées")
    st.dataframe(df[df[col_anomaly] == 1])
    st.subheader("📈 Visualisation")
    fig, ax = plt.subplots()
    ax.scatter(df['montant'], df['frequence'], c=df[col_anomaly], cmap='coolwarm', alpha=0.6)
    ax.set_xlabel("Montant")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Anomalies – {model_name}")
    st.pyplot(fig)
    csv = df.to_csv(index=False)
    st.download_button("📅 Télécharger CSV", csv, "transactions_resultats.csv", "text/csv")

def background():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://i.postimg.cc/jdDF3qcD/project.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9);
    }
    </style>
    """, unsafe_allow_html=True)

if st.session_state.page == "login":
    page_login()
elif st.session_state.page == "register":
    page_register()
elif st.session_state.page == "analyse":
    page_analyse()
