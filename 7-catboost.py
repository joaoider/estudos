# Etapa 1: Modelo Inicial com CatBoostRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Modelo inicial
cat_default = CatBoostRegressor(verbose=0, random_state=42)
cat_default.fit(X_train, y_train)

# Previsões
y_pred_cat_default = cat_default.predict(X_test)

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

avaliar_modelo("CatBoost - Padrão", y_test, y_pred_cat_default)
# ✅ Etapa 2: Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid_cat = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 5, 7],
    'l2_leaf_reg': [1, 3, 5],
    'bagging_temperature': [0, 1, 5]
}

cat_model = CatBoostRegressor(verbose=0, random_state=42)

grid_search_cat = GridSearchCV(estimator=cat_model,
                               param_grid=param_grid_cat,
                               scoring='neg_root_mean_squared_error',
                               cv=5,
                               n_jobs=-1,
                               verbose=2)

grid_search_cat.fit(X_train, y_train)

# ✅ Etapa 3: Avaliação do Melhor Modelo
# Melhor modelo
best_cat = grid_search_cat.best_estimator_
y_pred_cat_best = best_cat.predict(X_test)

print("Melhores parâmetros encontrados (CatBoost):")
print(grid_search_cat.best_params_)

avaliar_modelo("CatBoost - Otimizado", y_test, y_pred_cat_best)

# ✅ Etapa 4: Visualização – Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_cat_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("CatBoost Otimizado: Real vs Previsto")
plt.show()

# ✅ Etapa 5: Importância das Variáveis
import pandas as pd

importancias = pd.Series(best_cat.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Importância das Variáveis - CatBoost")
plt.show()