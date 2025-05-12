# main.py
import streamlit as st
from sklearn.datasets import fetch_california_housing
from eda import (
    analise_inicial,
    analise_visual,
    pre_processamento
)
from modelo_regressao_linear import rodar_regressao_linear
from modelo_random_forest import rodar_random_forest
from modelo_gradient_boosting import rodar_gradient_boosting
from modelo_xgboost import rodar_xgboost
from modelo_lightgbm import rodar_lightgbm
from modelo_catboost import rodar_catboost
from modelo_svr import rodar_svr
from modelo_regressoes_lineares_reg import rodar_regressoes_lineares
from modelo_knn import rodar_knn
from modelo_hist_gradient_boosting import rodar_hist_gradient_boosting

st.title("Análise de Regressão - Housing California")

# Carregar base
data = fetch_california_housing(as_frame=True)
df = data.frame
st.write("## Dados carregados")
st.dataframe(df.head())

# Etapas
analise_inicial(df, st)
analise_visual(df, st)
X_train, X_test, y_train, y_test, df_tratado = pre_processamento(df, st)

features = df_tratado.drop(columns=['MedHouseVal'])
rodar_regressao_linear(X_train, X_test, y_train, y_test, features, st)
rodar_random_forest(X_train, X_test, y_train, y_test, features, st)
rodar_gradient_boosting(X_train, X_test, y_train, y_test, X, st)
rodar_xgboost(X_train, X_test, y_train, y_test, features, st)
rodar_lightgbm(X_train, X_test, y_train, y_test, X, st)
rodar_catboost(X_train, X_test, y_train, y_test, X, st)
rodar_svr(X_train, X_test, y_train, y_test, X, st)
rodar_regressoes_lineares(X_train, X_test, y_train, y_test, X, st)
rodar_knn(X_train, X_test, y_train, y_test, features, st)
rodar_hist_gradient_boosting(X_train, X_test, y_train, y_test, features, st)