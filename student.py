import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def load_data(path='StudentsPerformance.csv'):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.dropna()

    df['average'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['passed'] = (df['average'] >= 60).astype(int)

    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def visualize_data(df):
    numeric_df = df.select_dtypes(include=['number'])
    st.subheader("📈 Matrice de corrélation (variables numériques)")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
