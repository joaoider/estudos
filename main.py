# main.py
import streamlit as st
from sklearn.datasets import fetch_california_housing

resultados = []

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

with st.sidebar.expander("📊 EDA"):
    mostrar_eda = st.checkbox("Exibir análise EDA")

with st.sidebar.expander("🤖 Modelos"):
    modelo_escolhido = st.radio(
        "Escolha o Modelo:",
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
        ],
        key="modelos_radio"
    )

with st.sidebar.expander("🏆 Melhor Modelo"):
    mostrar_melhor = st.checkbox("Comparar todos os modelos")


# Carregar base
data = fetch_california_housing(as_frame=True)
df = data.frame

# EDA
if mostrar_eda:
    st.subheader("📊 Análise Exploratória de Dados (EDA)")
    st.write("## Dados carregados")
    st.dataframe(df.head())

    analise_inicial(df, st)
    analise_visual(df, st)

# Pré-processamento (único)
X_train, X_test, y_train, y_test, df_tratado = pre_processamento(df, st)
features = df_tratado.drop(columns=['MedHouseVal'])

# Modelos
if modelo_escolhido:
    st.subheader(f"🤖 Resultado: {modelo_escolhido}")

    if modelo_escolhido == "Regressão Linear":
        resultado = rodar_regressao_linear(X_train, X_test, y_train, y_test, features, st)
    elif modelo_escolhido == "Random Forest":
        resultado = rodar_random_forest(X_train, X_test, y_train, y_test, features, st)
    # ... continue com os outros modelos ...

    resultados.append(resultado)

# Comparativo
if mostrar_melhor:
    if not resultados:
        st.warning("⚠️ Execute ao menos um modelo para comparar.")
    else:
        df_resultados = pd.DataFrame(resultados)
        st.subheader("📊 Comparativo de Modelos")
        st.dataframe(df_resultados)

        melhor = df_resultados.loc[df_resultados["RMSE"].idxmin()]
        st.markdown(f"""
        ### 🥇 Melhor Modelo: `{melhor['Modelo']}`
        - **MAE:** {melhor['MAE']:.3f}  
        - **RMSE:** {melhor['RMSE']:.3f}  
        - **R²:** {melhor['R²']:.3f}  
        - **Parâmetros:** `{melhor['Parâmetros']}`
        """)
