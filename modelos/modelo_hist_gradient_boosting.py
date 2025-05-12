# modelo_hist_gradient_boosting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def rodar_hist_gradient_boosting(X_train, X_test, y_train, y_test, X, st):
    st.subheader("📊 Modelo: HistGradientBoosting Regressor")

    # Etapa 1: Modelo inicial
    hgb_default = HistGradientBoostingRegressor(random_state=42)
    hgb_default.fit(X_train, y_train)
    y_pred_default = hgb_default.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_default)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_default))
    r2 = r2_score(y_test, y_pred_default)

    st.markdown("### 🎯 Avaliação do modelo padrão")
    st.write(f"**MAE:**  {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**R²:**   {r2:.3f}")

    # Etapa 2: Otimização com GridSearchCV
    st.markdown("### 🔬 Otimização com GridSearchCV")

    param_grid_hgb = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [20, 30, 50],
        'l2_regularization': [0.0, 0.1, 1.0],
    }

    hgb_model = HistGradientBoostingRegressor(random_state=42)

    grid_search_hgb = GridSearchCV(estimator=hgb_model,
                                   param_grid=param_grid_hgb,
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=0)

    grid_search_hgb.fit(X_train, y_train)

    best_hgb = grid_search_hgb.best_estimator_
    y_pred_best = best_hgb.predict(X_test)

    mae_best = mean_absolute_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)

    st.write("**Melhores parâmetros encontrados:**", grid_search_hgb.best_params_)

    st.markdown("### 📈 Avaliação do Modelo Otimizado")
    st.write(f"**MAE:**  {mae_best:.3f}")
    st.write(f"**RMSE:** {rmse_best:.3f}")
    st.write(f"**R²:**   {r2_best:.3f}")

    # Etapa 4: Visualização – Real vs Previsto
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred_best, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.set_title("HistGradientBoosting Otimizado: Real vs Previsto")
    st.pyplot(fig)

    return {
    "Modelo": "HistGradientBoosting",
    "MAE": mae_best,
    "RMSE": rmse_best,
    "R²": r2_best,
    "Parâmetros": grid_search_hgb.best_params_
}
