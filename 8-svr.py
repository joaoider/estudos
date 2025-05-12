# ✅ Etapa 1: Normalização dos dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ✅ Etapa 2: Modelo inicial com SVR
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Modelo padrão
svr_default = SVR()
svr_default.fit(X_train_scaled, y_train)
y_pred_svr_default = svr_default.predict(X_test_scaled)

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

avaliar_modelo("SVR - Padrão", y_test, y_pred_svr_default)
# ✅ Etapa 3: Otimização com GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid_svr = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 0.01, 0.1, 1]
}

svr_model = SVR()

grid_search_svr = GridSearchCV(estimator=svr_model,
                               param_grid=param_grid_svr,
                               scoring='neg_root_mean_squared_error',
                               cv=5,
                               n_jobs=-1,
                               verbose=2)

grid_search_svr.fit(X_train_scaled, y_train)
# ✅ Etapa 4: Avaliação do Melhor Modelo
# Melhor modelo
best_svr = grid_search_svr.best_estimator_
y_pred_svr_best = best_svr.predict(X_test_scaled)

print("Melhores parâmetros encontrados (SVR):")
print(grid_search_svr.best_params_)

avaliar_modelo("SVR - Otimizado", y_test, y_pred_svr_best)
# ✅ Etapa 5: Visualização – Real vs Previsto
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_svr_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("SVR Otimizado: Real vs Previsto")
plt.show()

