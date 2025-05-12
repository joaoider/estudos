# main.py
import streamlit as st
from sklearn.datasets import fetch_california_housing

# Importações EDA
from eda import (
    analise_inicial,
    analise_visual,
    pre_processamento
)

# Importações Modelos
from modelos.modelo_regressao_linear import rodar_regressao_linear
from modelos.modelo_random_forest import rodar_random_forest
from modelos.modelo_gradient_boosting import rodar_gradient_boosting
from modelos.modelo_xgboost import rodar_xgboost
from modelos.modelo_lightgbm import rodar_lightgbm
from modelos.modelo_catboost import rodar_catboost
from modelos.modelo_svr import rodar_svr
from modelos.modelo_regressoes_lineares_reg import rodar_regressoes_lineares
from modelos.modelo_knn import rodar_knn
from modelos.modelo_hist_gradient_boosting import rodar_hist_gradient_boosting

# Título da aplicação
st.title("🏡 Análise de Regressão - Housing California")

# Sidebar principal
secao = st.sidebar.selectbox("🔎 Escolha a Seção:", ["EDA", "Modelos"])

# Carregar base
data = fetch_california_housing(as_frame=True)
df = data.frame

# EDA
if secao == "EDA":
    st.subheader("📊 Análise Exploratória de Dados (EDA)")
    st.write("## Dados carregados")
    st.dataframe(df.head())

    analise_inicial(df, st)
    analise_visual(df, st)

# Modelos
elif secao == "Modelos":
    modelo_escolhido = st.sidebar.radio(
        "📌 Escolha o Modelo:",
        [
            "Regressão Linear",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "SVR",
            "Regressões com Regularização (Ridge, Lasso, ElasticNet)",
            "KNN",
            "HistGradientBoosting"
        ]
    )

    # Pré-processamento (roda apenas uma vez)
    X_train, X_test, y_train, y_test, df_tratado = pre_processamento(df, st)
    features = df_tratado.drop(columns=['MedHouseVal'])

    # Roteamento de modelo
    if modelo_escolhido == "Regressão Linear":
        rodar_regressao_linear(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "Random Forest":
        rodar_random_forest(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "Gradient Boosting":
        rodar_gradient_boosting(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "XGBoost":
        rodar_xgboost(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "LightGBM":
        rodar_lightgbm(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "CatBoost":
        rodar_catboost(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "SVR":
        rodar_svr(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "Regressões com Regularização (Ridge, Lasso, ElasticNet)":
        rodar_regressoes_lineares(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "KNN":
        rodar_knn(X_train, X_test, y_train, y_test, features, st)

    elif modelo_escolhido == "HistGradientBoosting":
        rodar_hist_gradient_boosting(X_train, X_test, y_train, y_test, features, st)