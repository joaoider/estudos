# Modelo Inicial com LGBMRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Modelo padrão
lgb_default = LGBMRegressor(random_state=42)
lgb_default.fit(X_train, y_train)

# Previsão
y_pred_lgb_default = lgb_default.predict(X_test)

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

avaliar_modelo("LightGBM - Padrão", y_test, y_pred_lgb_default)


# Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

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
                                verbose=2)

grid_search_lgb.fit(X_train, y_train)


# Avaliação do Melhor Modelo
# Melhor modelo treinado
best_lgb = grid_search_lgb.best_estimator_
y_pred_lgb_best = best_lgb.predict(X_test)

print("Melhores parâmetros encontrados (LightGBM):")
print(grid_search_lgb.best_params_)

avaliar_modelo("LightGBM - Otimizado", y_test, y_pred_lgb_best)


# Visualização – Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lgb_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("LightGBM Otimizado: Real vs Previsto")
plt.show()

# Importância das Variáveis
import pandas as pd

importancias = pd.Series(best_lgb.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Importância das Variáveis - LightGBM")
plt.show()