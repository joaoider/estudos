# modelo_random_forest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

def rodar_random_forest(X_train, X_test, y_train, y_test, X, st):
    st.subheader("🌲 Modelo: Random Forest Regressor")

    # Modelo padrão
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    st.markdown("### 🎯 Avaliação do modelo padrão")
    st.write(f"**MAE:**  {mae_rf:.3f}")
    st.write(f"**RMSE:** {rmse_rf:.3f}")
    st.write(f"**R²:**   {r2_rf:.3f}")

    # Importância das variáveis
    st.markdown("### 🔍 Importância das Variáveis")
    importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(10,6))
    importancias.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("Importância das Variáveis - Random Forest")
    st.pyplot(fig)

    # Variação de n_estimators
    st.markdown("### 📈 Variação de n_estimators")
    n_estimators_range = [10, 50, 100, 200, 300]
    r2_scores, rmse_scores = [], []

    for n in n_estimators_range:
        rf = RandomForestRegressor(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(n_estimators_range, r2_scores, marker='o')
    ax[0].set_title("R² vs Número de Árvores")
    ax[0].set_xlabel("n_estimators")
    ax[0].set_ylabel("R²")

    ax[1].plot(n_estimators_range, rmse_scores, marker='o', color='orange')
    ax[1].set_title("RMSE vs Número de Árvores")
    ax[1].set_xlabel("n_estimators")
    ax[1].set_ylabel("RMSE")

    st.pyplot(plt.gcf())

    # Grid Search
    st.markdown("### 🔬 GridSearchCV")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42),
                        param_grid, cv=5,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    st.write("**Melhores parâmetros (GridSearchCV):**", grid.best_params_)
    st.write("**Melhor RMSE (negativo):**", grid.best_score_)

    # Randomized Search
    st.markdown("### 🎲 RandomizedSearchCV")
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(5, 21)),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5)
    }

    random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                                       param_distributions=param_dist,
                                       n_iter=20, cv=5,
                                       scoring='neg_root_mean_squared_error',
                                       n_jobs=-1, verbose=0,
                                       random_state=42)
    random_search.fit(X_train, y_train)
    st.write("**Melhores parâmetros (RandomizedSearchCV):**", random_search.best_params_)
    st.write("**Melhor RMSE (negativo):**", random_search.best_score_)

    # Previsões com os melhores modelos
    best_grid_model = grid.best_estimator_
    best_random_model = random_search.best_estimator_

    y_pred_grid = best_grid_model.predict(X_test)
    y_pred_random = best_random_model.predict(X_test)

    def avaliar(nome, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.markdown(f"**{nome}**")
        st.write(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    st.markdown("### 📊 Avaliação dos modelos otimizados")
    avaliar("GridSearchCV", y_pred_grid)
    avaliar("RandomizedSearchCV", y_pred_random)

    # Gráficos de dispersão
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_grid, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_title("GridSearchCV: Real vs Previsto")
    axes[0].set_xlabel("Valor Real")
    axes[0].set_ylabel("Valor Previsto")

    axes[1].scatter(y_test, y_pred_random, alpha=0.5, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1].set_title("RandomizedSearchCV: Real vs Previsto")
    axes[1].set_xlabel("Valor Real")
    axes[1].set_ylabel("Valor Previsto")

    st.pyplot(fig)

    return {
    "Modelo": "Regressão Linear",
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2,
    "Parâmetros": "default"
}