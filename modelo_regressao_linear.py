# modelo_regressao_linear.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def rodar_regressao_linear(X_train, X_test, y_train, y_test, X, st):
    st.subheader("📈 Regressão Linear")

    # Treinamento
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### Métricas do Modelo")
    st.write(f"**MAE:**  {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**R²:**   {r2:.3f}")

    # Gráfico Real vs Previsto
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.4)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.set_title("Regressão Linear: Real vs. Previsto")
    st.pyplot(fig)

    # Resíduos
    residuos = y_test - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuos, alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Valores Previstos")
    ax.set_ylabel("Resíduos")
    ax.set_title("Resíduos vs Valores Previstos")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(residuos, kde=True, ax=ax)
    ax.set_title("Distribuição dos Resíduos")
    st.pyplot(fig)

    # VIF
    st.markdown("### VIF (Multicolinearidade)")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)

    # MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    st.write(f"**MAPE:** {mape:.2f}%")

    # Análise estatística com statsmodels
    st.markdown("### Resumo Estatístico (statsmodels)")
    X_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_sm).fit()
    st.text(model_sm.summary())