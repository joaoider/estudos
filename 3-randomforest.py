# Treinando o modelo
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Avaliação
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"📊 Resultados do Random Forest:")
print(f"MAE:  {mae_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")
print(f"R²:   {r2_rf:.3f}")

# Importância das variáveis
import pandas as pd
import matplotlib.pyplot as plt
importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Importância das Variáveis - Random Forest")
plt.show()

# Exemplo: Variação de n_estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

n_estimators_range = [10, 50, 100, 200, 300]
r2_scores = []
rmse_scores = []

for n in n_estimators_range:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    r2_scores.append(r2)
    rmse_scores.append(rmse)

# Gráfico de desempenho
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, r2_scores, marker='o')
plt.title("R² vs Número de Árvores")
plt.xlabel("n_estimators")
plt.ylabel("R²")

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, rmse_scores, marker='o', color='orange')
plt.title("RMSE vs Número de Árvores")
plt.xlabel("n_estimators")
plt.ylabel("RMSE")

plt.tight_layout()
plt.show()


# GridSearchCV com RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Dicionário de parâmetros para busca exaustiva
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,                 # 5-fold cross-validation
                           scoring='neg_root_mean_squared_error',  # usar RMSE como métrica
                           n_jobs=-1,            # usar todos os núcleos da CPU
                           verbose=2)

grid_search.fit(X_train, y_train)

print("Melhores parâmetros (GridSearchCV):", grid_search.best_params_)
print("Melhor RMSE (negativo):", grid_search.best_score_)


# RandomizedSearchCV com RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Faixas aleatórias de busca
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(5, 21)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=20,       # número de combinações testadas
                                   cv=5,
                                   scoring='neg_root_mean_squared_error',
                                   n_jobs=-1,
                                   verbose=2,
                                   random_state=42)

random_search.fit(X_train, y_train)

print("Melhores parâmetros (RandomizedSearchCV):", random_search.best_params_)
print("Melhor RMSE (negativo):", random_search.best_score_)


# Previsões com os melhores modelos
# Previsões com o melhor modelo do GridSearchCV
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)

# Previsões com o melhor modelo do RandomizedSearchCV
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)


# Avaliação de desempenho
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def avaliar_modelo(nome, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"📈 {nome}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print("-" * 40)

avaliar_modelo("GridSearchCV", y_test, y_pred_grid)
avaliar_modelo("RandomizedSearchCV", y_test, y_pred_random)


# Visualização comparativa
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Subplot para GridSearchCV
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_grid, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("GridSearchCV: Valores Reais vs Previstos")
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")

# Subplot para RandomizedSearchCV
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_random, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("RandomizedSearchCV: Valores Reais vs Previstos")
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")

plt.tight_layout()
plt.show()