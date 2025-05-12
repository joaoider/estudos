# modelo_gradient_boosting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def rodar_gradient_boosting(X_train, X_test, y_train, y_test, X, st):
    st.subheader("🌟 Modelo: Gradient Boosting Regressor")

    # Modelo inicial padrão
    gb_default = GradientBoostingRegressor(random_state=42)
    gb_default.fit(X_train, y_train)
    y_pred_default = gb_default.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_default)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_default))
    r2 = r2_score(y_test, y_pred_default)

    st.markdown("### 🎯 Avaliação do modelo padrão")
    st.write(f"**MAE:**  {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**R²:**   {r2:.3f}")

    # Otimização com GridSearchCV
    st.markdown("### 🔬 Otimização com GridSearchCV")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 1.0]
    }

    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=gb_model,
                               param_grid=param_grid,
                               cv=5,
                               scoring='neg_root_mean_squared_error',
                               n_jobs=-1,
                               verbose=0)

    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_
    y_pred_best = best_gb.predict(X_test)

    st.write("**Melhores parâmetros:**", grid_search.best_params_)

    mae_opt = mean_absolute_error(y_test, y_pred_best)
    rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_opt = r2_score(y_test, y_pred_best)

    st.markdown("### 📈 Avaliação do Modelo Otimizado")
    st.write(f"**MAE:**  {mae_opt:.3f}")
    st.write(f"**RMSE:** {rmse_opt:.3f}")
    st.write(f"**R²:**   {r2_opt:.3f}")

    # Gráfico de Real vs Previsto
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred_best, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.set_title("Gradient Boosting Otimizado: Real vs Previsto")
    st.pyplot(fig)

    # Importância das Variáveis
    st.markdown("### 🔍 Importância das Variáveis")
    importancias = pd.Series(best_gb.feature_importances_, index=X.columns)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    importancias.sort_values().plot(kind='barh', ax=ax2)
    ax2.set_title("Importância das Variáveis - Gradient Boosting")
    st.pyplot(fig2)