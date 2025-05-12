# eda_preprocess.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def analise_inicial(df, st):
    st.subheader("🔍 Primeiras Informações")
    st.write("**Primeiras linhas:**")
    st.dataframe(df.head())

    st.write("**Info:**")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("**Estatísticas descritivas:**")
    st.dataframe(df.describe())

    st.write(f"**Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")
    st.write("**Valores nulos:**")
    st.write(df.isnull().sum())

def analise_visual(df, st):
    st.subheader("📊 Visualização dos Dados")

    st.write("**Distribuição das variáveis**")
    fig, ax = plt.subplots(figsize=(15, 10))
    df.hist(bins=30, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("**Mapa de Correlação**")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.write("**Renda Média vs Valor Médio da Casa**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', ax=ax)
    st.pyplot(fig)

    st.write("**Média de Quartos vs Valor das Casas**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='AveRooms', y='MedHouseVal', ax=ax)
    st.pyplot(fig)

    st.write("**Visualização geográfica: Latitude × Longitude**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Longitude', y='Latitude',
                    hue='MedHouseVal', palette='viridis', alpha=0.7, ax=ax)
    st.pyplot(fig)

def pre_processamento(df, st):
    st.subheader("⚙️ Pré-processamento dos Dados")

    st.write("**Boxplots para verificação de outliers**")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(df.columns[:-1]):
        sns.boxplot(x=df[col], ax=axes[i//3][i%3])
        axes[i//3][i%3].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    # Transformações
    df['MedInc_log'] = np.log1p(df['MedInc'])
    df['RendaPorPessoa'] = df['MedInc'] / df['AveOccup']

    # Normalização
    features = df.drop(columns=['MedHouseVal'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X = pd.DataFrame(X_scaled, columns=features.columns)
    y = df['MedHouseVal']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    st.success("Pré-processamento concluído.")
    return X_train, X_test, y_train, y_test, df