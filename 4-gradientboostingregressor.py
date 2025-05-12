# Modelo inicial – Gradient Boosting Padrão
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Modelo inicial com parâmetros padrão
gb_default = GradientBoostingRegressor(random_state=42)
gb_default.fit(X_train, y_train)

# Previsões
y_pred_default = gb_default.predict(X_test)

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

avaliar_modelo("GradientBoosting - Padrão", y_test, y_pred_default)

# Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

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
                           verbose=2)

grid_search.fit(X_train, y_train)

# Avaliação do melhor modelo
# Melhor modelo
best_gb = grid_search.best_estimator_
y_pred_best = best_gb.predict(X_test)

print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

avaliar_modelo("GradientBoosting - Otimizado", y_test, y_pred_best)


# Visualização – Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("Gradient Boosting Otimizado: Real vs Previsto")
plt.show()

# Importância das Variáveis
import pandas as pd
importancias = pd.Series(best_gb.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Importância das Variáveis - Gradient Boosting")
plt.show()