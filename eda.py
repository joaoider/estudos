# eda_preprocess.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def analise_inicial(df):
    print("🔍 Primeiras linhas:")
    print(df.head())
    print("\n📋 Info:")
    print(df.info())
    print("\n📊 Estatísticas descritivas:")
    print(df.describe())
    print(f"\n🧾 Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print("\n❓ Valores nulos:")
    print(df.isnull().sum())

def analise_visual(df):
    # Distribuições
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle('Distribuições das Variáveis', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Correlação entre Variáveis')
    plt.show()

    # Scatterplots
    sns.scatterplot(data=df, x='MedInc', y='MedHouseVal')
    plt.title('Renda Média vs Valor Médio da Casa')
    plt.show()

    sns.scatterplot(data=df, x='AveRooms', y='MedHouseVal')
    plt.title('Média de Quartos vs Valor das Casas')
    plt.show()

    # Visualização geográfica
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Longitude', y='Latitude',
                    hue='MedHouseVal', palette='viridis', alpha=0.7)
    plt.title('Distribuição Geográfica dos Valores das Casas na Califórnia')
    plt.legend(title='Valor Médio da Casa')
    plt.show()

def pre_processamento(df):
    # Outliers (boxplots)
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.columns[:-1]):  # exceto MedHouseVal
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # Transformações (opcional)
    df['MedInc_log'] = np.log1p(df['MedInc'])

    # Engenharia de atributos
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
    return X_train, X_test, y_train, y_test, df