# modelo_regressoes_lineares_reg.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def rodar_regressoes_lineares(X_train, X_test, y_train, y_test, X, st):
    st.subheader("📐 Modelos Lineares com Regularização (Ridge, Lasso, ElasticNet)")

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def avaliar(nome, y_real, y_pred):
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)
        st.markdown(f"**{nome}**")
        st.write(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    # Modelos padrão
    st.markdown("### 🔹 Modelos Padrão")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    avaliar("Ridge", y_test, y_pred_ridge)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    avaliar("Lasso", y_test, y_pred_lasso)

    enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    enet.fit(X_train_scaled, y_train)
    y_pred_enet = enet.predict(X_test_scaled)
    avaliar("ElasticNet", y_test, y_pred_enet)

    # Otimização com GridSearchCV
    st.markdown("### 🔬 Otimização com GridSearchCV")

    # Ridge
    param_grid_ridge = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_ridge = GridSearchCV(Ridge(), param_grid_ridge, scoring='neg_root_mean_squared_error', cv=5)
    grid_ridge.fit(X_train_scaled, y_train)
    y_pred_grid_ridge = grid_ridge.best_estimator_.predict(X_test_scaled)
    st.write("**Melhor Ridge:**", grid_ridge.best_params_)
    avaliar("Ridge - Otimizado", y_test, y_pred_grid_ridge)

    # Lasso
    param_grid_lasso = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    grid_lasso = GridSearchCV(Lasso(max_iter=10000), param_grid_lasso,
                              scoring='neg_root_mean_squared_error', cv=5)
    grid_lasso.fit(X_train_scaled, y_train)
    y_pred_grid_lasso = grid_lasso.best_estimator_.predict(X_test_scaled)
    st.write("**Melhor Lasso:**", grid_lasso.best_params_)
    avaliar("Lasso - Otimizado", y_test, y_pred_grid_lasso)

    # ElasticNet
    param_grid_enet = {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    grid_enet = GridSearchCV(ElasticNet(max_iter=10000), param_grid_enet,
                             scoring='neg_root_mean_squared_error', cv=5)
    grid_enet.fit(X_train_scaled, y_train)
    y_pred_grid_enet = grid_enet.best_estimator_.predict(X_test_scaled)
    st.write("**Melhor ElasticNet:**", grid_enet.best_params_)
    avaliar("ElasticNet - Otimizado", y_test, y_pred_grid_enet)

    mae_ridge = mean_absolute_error(y_test, y_pred_grid_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_grid_ridge))
    r2_ridge = r2_score(y_test, y_pred_grid_ridge)

    mae_lasso = mean_absolute_error(y_test, y_pred_grid_lasso)
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_grid_lasso))
    r2_lasso = r2_score(y_test, y_pred_grid_lasso)

    mae_enet = mean_absolute_error(y_test, y_pred_grid_enet)
    rmse_enet = np.sqrt(mean_squared_error(y_test, y_pred_grid_enet))
    r2_enet = r2_score(y_test, y_pred_grid_enet)

    melhor_rmse = min(rmse_ridge, rmse_lasso, rmse_enet)

    if melhor_rmse == rmse_ridge:
        return {
        "Modelo": "Ridge",
        "MAE": mae_ridge,
        "RMSE": rmse_ridge,
        "R²": r2_ridge,
        "Parâmetros": grid_ridge.best_params_
    }
    elif melhor_rmse == rmse_lasso:
        return {
        "Modelo": "Lasso",
        "MAE": mae_lasso,
        "RMSE": rmse_lasso,
        "R²": r2_lasso,
        "Parâmetros": grid_lasso.best_params_
    }
    else:
        return {
        "Modelo": "ElasticNet",
        "MAE": mae_enet,
        "RMSE": rmse_enet,
        "R²": r2_enet,
        "Parâmetros": grid_enet.best_params_
    }
