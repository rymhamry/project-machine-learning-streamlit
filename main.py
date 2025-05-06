import streamlit as st
from student import load_data, preprocess_data, visualize_data
from classification import train_model, evaluate_model

st.title("📊 Analyse des Performances Étudiantes")

st.sidebar.header("Options")
if st.sidebar.button("Charger les données"):
    df_raw = load_data()
    st.write("Aperçu des données brutes", df_raw.head())

    df_clean = preprocess_data(df_raw)
    st.write("Données après prétraitement", df_clean.head())

    visualize_data(df_raw)

    model, X_test, y_test, y_pred, feature_importance_df = train_model(df_clean)
    report, matrix = evaluate_model(y_test, y_pred)

    st.subheader("📈 Évaluation du modèle")
    st.text("Rapport de classification :")
    st.dataframe(report)

    st.text("Matrice de confusion :")
    st.write(matrix)

    st.subheader("🔍 Importance des variables")
    st.dataframe(feature_importance_df)
