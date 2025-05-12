# modelo_svr.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def rodar_svr(X_train, X_test, y_train, y_test, X, st):
    st.subheader("🔷 Modelo: Suporte Vetorial para Regressão (SVR)")

    # Etapa 1: Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo padrão
    svr_default = SVR()
    svr_default.fit(X_train_scaled, y_train)
    y_pred_default = svr_default.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred_default)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_default))
    r2 = r2_score(y_test, y_pred_default)

    st.markdown("### 🎯 Avaliação do modelo padrão")
    st.write(f"**MAE:**  {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**R²:**   {r2:.3f}")

    # Otimização com GridSearchCV
    st.markdown("### 🔬 Otimização com GridSearchCV")

    param_grid_svr = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'gamma': ['scale', 0.01, 0.1, 1]
    }

    svr_model = SVR()

    grid_search_svr = GridSearchCV(estimator=svr_model,
                                   param_grid=param_grid_svr,
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=0)

    grid_search_svr.fit(X_train_scaled, y_train)

    best_svr = grid_search_svr.best_estimator_
    y_pred_best = best_svr.predict(X_test_scaled)

    mae_best = mean_absolute_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)

    st.write("**Melhores parâmetros encontrados:**", grid_search_svr.best_params_)
    st.markdown("### 📈 Avaliação do Modelo Otimizado")
    st.write(f"**MAE:**  {mae_best:.3f}")
    st.write(f"**RMSE:** {rmse_best:.3f}")
    st.write(f"**R²:**   {r2_best:.3f}")

    # Gráfico Real vs Previsto
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred_best, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.set_title("SVR Otimizado: Real vs Previsto")
    st.pyplot(fig)

    return {
    "Modelo": "SVR",
    "MAE": mae_best,
    "RMSE": rmse_best,
    "R²": r2_best,
    "Parâmetros": grid_search_svr.best_params_
}
