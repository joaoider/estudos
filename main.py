# main.py
import streamlit as st
from sklearn.datasets import fetch_california_housing
from eda import (
    analise_inicial,
    analise_visual,
    pre_processamento
)
from modelo_regressao_linear import rodar_regressao_linear

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