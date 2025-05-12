# Modelo Inicial com XGBRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Modelo inicial padrão
xgb_default = XGBRegressor(random_state=42, verbosity=0)
xgb_default.fit(X_train, y_train)

# Previsões
y_pred_xgb_default = xgb_default.predict(X_test)

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

avaliar_modelo("XGBoost - Padrão", y_test, y_pred_xgb_default)


# Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBRegressor(random_state=42, verbosity=0)

grid_search_xgb = GridSearchCV(estimator=xgb_model,
                                param_grid=param_grid_xgb,
                                scoring='neg_root_mean_squared_error',
                                cv=5,
                                n_jobs=-1,
                                verbose=2)

grid_search_xgb.fit(X_train, y_train)


# Avaliação do melhor modelo
# Melhor modelo
best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb_best = best_xgb.predict(X_test)

print("Melhores parâmetros encontrados (XGBoost):")
print(grid_search_xgb.best_params_)

avaliar_modelo("XGBoost - Otimizado", y_test, y_pred_xgb_best)

# Visualização - Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_xgb_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("XGBoost Otimizado: Real vs Previsto")
plt.show()


# Importância das Variáveis
import pandas as pd

importancias = pd.Series(best_xgb.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Importância das Variáveis - XGBoost")
plt.show()