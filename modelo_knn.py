# modelo_knn.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def rodar_knn(X_train, X_test, y_train, y_test, X, st):
    st.subheader("📍 Modelo: K-Nearest Neighbors (KNN)")

    # Etapa 1: Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Etapa 2: Modelo Padrão
    knn_default = KNeighborsRegressor(n_neighbors=5)
    knn_default.fit(X_train_scaled, y_train)
    y_pred_default = knn_default.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred_default)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_default))
    r2 = r2_score(y_test, y_pred_default)

    st.markdown("### 🎯 Avaliação do modelo padrão (k=5)")
    st.write(f"**MAE:**  {mae:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**R²:**   {r2:.3f}")

    # Etapa 3: GridSearchCV
    st.markdown("### 🔬 Otimização com GridSearchCV")

    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn_model = KNeighborsRegressor()
    grid_search_knn = GridSearchCV(knn_model,
                                   param_grid=param_grid_knn,
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=0)

    grid_search_knn.fit(X_train_scaled, y_train)
    best_knn = grid_search_knn.best_estimator_
    y_pred_best = best_knn.predict(X_test_scaled)

    mae_best = mean_absolute_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)

    st.write("**Melhores parâmetros encontrados:**", grid_search_knn.best_params_)
    st.markdown("### 📈 Avaliação do Modelo Otimizado")
    st.write(f"**MAE:**  {mae_best:.3f}")
    st.write(f"**RMSE:** {rmse_best:.3f}")
    st.write(f"**R²:**   {r2_best:.3f}")

    # Etapa 5: Visualização – Real vs Previsto
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred_best, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.set_title("KNN Regressor Otimizado: Real vs Previsto")
    st.pyplot(fig)