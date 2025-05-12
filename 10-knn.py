# ✅ Etapa 1: Normalização dos dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ✅ Etapa 2: Modelo inicial com KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Modelo KNN padrão
knn_default = KNeighborsRegressor(n_neighbors=5)
knn_default.fit(X_train_scaled, y_train)
y_pred_knn_default = knn_default.predict(X_test_scaled)

def avaliar_modelo(nome, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"📈 {nome}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print("-" * 40)

avaliar_modelo("KNN - Padrão (k=5)", y_test, y_pred_knn_default)
# ✅ Etapa 3: Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

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
                               verbose=2)

grid_search_knn.fit(X_train_scaled, y_train)
# ✅ Etapa 4: Avaliação do Melhor Modelo

# Melhor modelo encontrado
best_knn = grid_search_knn.best_estimator_
y_pred_knn_best = best_knn.predict(X_test_scaled)

print("Melhores parâmetros encontrados (KNN):")
print(grid_search_knn.best_params_)

avaliar_modelo("KNN - Otimizado", y_test, y_pred_knn_best)

# ✅ Etapa 5: Visualização – Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_knn_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("KNN Regressor Otimizado: Real vs Previsto")
plt.show()