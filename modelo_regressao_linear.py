# modelo_regressao_linear.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def rodar_regressao_linear(X_train, X_test, y_train, y_test, X):
    print("\n=== Regressão Linear ===")
    # Treinamento
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Resultados:")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    # Gráfico Real vs Previsto
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Regressão Linear: Real vs. Previsto")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

    # Resíduos
    residuos = y_test - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Valores Previstos")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Previstos")
    plt.show()

    sns.histplot(residuos, kde=True)
    plt.title("Distribuição dos Resíduos")
    plt.show()

    # VIF
    print("\n📌 VIF (Multicolinearidade):")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)

    # MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"\n📉 MAPE: {mape:.2f}%")

    # Análise estatística com statsmodels
    X_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_sm).fit()
    print("\n📈 Resumo estatístico (statsmodels):")
    print(model_sm.summary())