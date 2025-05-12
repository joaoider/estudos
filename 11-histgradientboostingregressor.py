# Etapa 1: Modelo inicial com HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Modelo padrão
hgb_default = HistGradientBoostingRegressor(random_state=42)
hgb_default.fit(X_train, y_train)
y_pred_hgb_default = hgb_default.predict(X_test)

# Avaliação
def avaliar_modelo(nome, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"📈 {nome}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print("-" * 40)

avaliar_modelo("HistGradientBoosting - Padrão", y_test, y_pred_hgb_default)
# ✅ Etapa 2: Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

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
                               verbose=2)

grid_search_hgb.fit(X_train, y_train)
# ✅ Etapa 3: Avaliação do Melhor Modelo
best_hgb = grid_search_hgb.best_estimator_
y_pred_hgb_best = best_hgb.predict(X_test)

print("Melhores parâmetros encontrados (HistGradientBoosting):")
print(grid_search_hgb.best_params_)

avaliar_modelo("HistGradientBoosting - Otimizado", y_test, y_pred_hgb_best)
# ✅ Etapa 4: Visualização – Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_hgb_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("HistGradientBoosting Otimizado: Real vs Previsto")
plt.show()