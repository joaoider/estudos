# ✅ Etapa 1: Normalização (essencial)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ✅ Etapa 2: Modelos iniciais
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
avaliar_modelo("Ridge", y_test, y_pred_ridge)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
avaliar_modelo("Lasso", y_test, y_pred_lasso)

# ElasticNet
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train_scaled, y_train)
y_pred_enet = enet.predict(X_test_scaled)
avaliar_modelo("ElasticNet", y_test, y_pred_enet)
✅ Etapa 3: Otimização com GridSearchCV para cada modelo
# Ridge (L2):
from sklearn.model_selection import GridSearchCV

param_grid_ridge = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_model = Ridge()
grid_ridge = GridSearchCV(ridge_model, param_grid_ridge, scoring='neg_root_mean_squared_error', cv=5)
grid_ridge.fit(X_train_scaled, y_train)
print("Melhor Ridge:", grid_ridge.best_params_)
avaliar_modelo("Ridge - Otimizado", y_test, grid_ridge.best_estimator_.predict(X_test_scaled))
# Lasso (L1):
param_grid_lasso = {'alpha': [0.01, 0.1, 1.0, 10.0]}
lasso_model = Lasso(max_iter=10000)
grid_lasso = GridSearchCV(lasso_model, param_grid_lasso, scoring='neg_root_mean_squared_error', cv=5)
grid_lasso.fit(X_train_scaled, y_train)
print("Melhor Lasso:", grid_lasso.best_params_)
avaliar_modelo("Lasso - Otimizado", y_test, grid_lasso.best_estimator_.predict(X_test_scaled))
# ElasticNet:
param_grid_enet = {
    'alpha': [0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}
enet_model = ElasticNet(max_iter=10000)
grid_enet = GridSearchCV(enet_model, param_grid_enet, scoring='neg_root_mean_squared_error', cv=5)
grid_enet.fit(X_train_scaled, y_train)
print("Melhor ElasticNet:", grid_enet.best_params_)
avaliar_modelo("ElasticNet - Otimizado", y_test, grid_enet.best_estimator_.predict(X_test_scaled))