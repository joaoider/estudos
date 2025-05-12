# modelo_lightgbm.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def rodar_lightgbm(X_train, X_test, y_train, y_test, X, st):
    st.subheader("🌿 Modelo: LightGBM Regressor")

    # Modelo inicial
    lgb_default = LGBMRegressor(random_state=42)
    lgb_default.fit(X_train, y_train)
    y_pred_default = lgb_default.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_default)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_default))
    r2 = r2_score(y_test, y_pred_default)

    st.markdown("### 🎯 Avaliação do modelo padrão")
    st.write(f"**MAE:**  {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**R²:**   {r2:.3f}")

    # GridSearchCV
    st.markdown("### 🔬 Otimização com GridSearchCV")

    param_grid_lgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [20, 31, 40],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    lgb_model = LGBMRegressor(random_state=42)

    grid_search_lgb = GridSearchCV(estimator=lgb_model,
                                   param_grid=param_grid_lgb,
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=0)

    grid_search_lgb.fit(X_train, y_train)

    best_lgb = grid_search_lgb.best_estimator_
    y_pred_best = best_lgb.predict(X_test)

    mae_best = mean_absolute_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)

    st.write("**Melhores parâmetros:**", grid_search_lgb.best_params_)

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
    ax.set_title("LightGBM Otimizado: Real vs Previsto")
    st.pyplot(fig)

    # Importância das Variáveis
    st.markdown("### 🔍 Importância das Variáveis")
    importancias = pd.Series(best_lgb.feature_importances_, index=X.columns)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    importancias.sort_values().plot(kind='barh', ax=ax2)
    ax2.set_title("Importância das Variáveis - LightGBM")
    st.pyplot(fig2)

    return {
    "Modelo": "Regressão Linear",
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2,
    "Parâmetros": "default"
}