# 🏡 California Housing - Análise de Regressão com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descrição do Projeto

Este projeto implementa uma aplicação completa de **Machine Learning para Regressão** utilizando o dataset **California Housing** do scikit-learn. A aplicação oferece uma interface interativa via Streamlit para análise exploratória de dados (EDA), pré-processamento e comparação de múltiplos algoritmos de regressão.

## 🎯 Objetivos

- **Análise Exploratória de Dados (EDA)** completa e visual
- **Pré-processamento automático** de dados
- **Implementação profissional** de 10 algoritmos de regressão
- **Otimização de hiperparâmetros** com GridSearch e RandomizedSearch
- **Comparação sistemática** de performance entre modelos
- **Interface intuitiva** para usuários não técnicos

## 🚀 Funcionalidades

### 📊 Análise Exploratória de Dados (EDA)
- Estatísticas descritivas completas
- Análise de correlação entre variáveis
- Visualizações interativas (histogramas, boxplots, scatter plots)
- Detecção automática de outliers e valores faltantes

### 🤖 Algoritmos de Regressão Implementados

| Algoritmo | Status | Otimização | Visualizações |
|-----------|--------|------------|---------------|
| **Random Forest** | ✅ Profissional | GridSearch + RandomizedSearch | Feature Importance, Hyperparameter Analysis |
| **XGBoost** | ✅ Implementado | GridSearch | Learning Curves, Feature Importance |
| **LightGBM** | ✅ Implementado | GridSearch | Learning Curves, Feature Importance |
| **CatBoost** | ✅ Implementado | GridSearch | Learning Curves, Feature Importance |
| **Gradient Boosting** | ✅ Implementado | GridSearch | Learning Curves, Feature Importance |
| **SVR (Support Vector Regression)** | ✅ Implementado | GridSearch | Kernel Analysis |
| **Regressão Linear** | ✅ Implementado | Ridge/Lasso/ElasticNet | Residual Analysis |
| **KNN** | ✅ Implementado | GridSearch | K-value Analysis |
| **HistGradientBoosting** | ✅ Implementado | GridSearch | Learning Curves |

### 🔧 Otimização de Hiperparâmetros
- **GridSearchCV**: Busca exaustiva em grade de parâmetros
- **RandomizedSearchCV**: Busca aleatória eficiente
- **Cross-validation**: Validação cruzada com 5 folds
- **Métricas múltiplas**: MAE, RMSE, R²

## 🛠️ Tecnologias Utilizadas

### Core ML
- **Scikit-learn**: Algoritmos de machine learning
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualizações

### Interface e Deploy
- **Streamlit**: Interface web interativa
- **Plotly**: Gráficos interativos avançados

### Otimização
- **GridSearchCV**: Otimização de hiperparâmetros
- **RandomizedSearchCV**: Busca eficiente de parâmetros

## 📁 Estrutura do Projeto

```
estudos/
├── 📁 modelos/                          # Implementações dos algoritmos
│   ├── 🆕 modelo_random_forest.py      # Random Forest profissional
│   ├── modelo_xgboost.py               # XGBoost
│   ├── modelo_lightgbm.py             # LightGBM
│   ├── modelo_catboost.py             # CatBoost
│   ├── modelo_gradient_boosting.py    # Gradient Boosting
│   ├── modelo_svr.py                  # Support Vector Regression
│   ├── modelo_regressao_linear.py     # Regressão Linear
│   ├── modelo_regressoes_lineares_reg.py # Ridge/Lasso/ElasticNet
│   ├── modelo_knn.py                  # K-Nearest Neighbors
│   └── modelo_hist_gradient_boosting.py # Histogram-based GB
├── 📄 main.py                          # Aplicação principal Streamlit
├── 📄 eda.py                           # Análise exploratória de dados
├── 📄 datasets.py                      # Gerenciamento de datasets
├── 📄 requirements.txt                 # Dependências do projeto
└── 📄 README.md                        # Este arquivo
```

## 🚀 Como Executar

### 1. Pré-requisitos
```bash
Python 3.8+
pip ou conda
```

### 2. Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/estudos.git
cd estudos

# Instale as dependências
pip install -r requirements.txt
```

### 3. Execução
```bash
# Execute a aplicação Streamlit
streamlit run main.py
```

A aplicação estará disponível em: `http://localhost:8501`

## 📖 Como Usar

### 1. **Análise Exploratória (EDA)**
- Marque a checkbox "Exibir análise EDA" na sidebar
- Visualize estatísticas descritivas e correlações
- Analise distribuições e outliers

### 2. **Seleção de Modelo**
- Escolha um algoritmo na seção "Modelos"
- Cada modelo executa automaticamente:
  - Treinamento com dados de teste
  - Otimização de hiperparâmetros
  - Avaliação com métricas múltiplas
  - Visualizações específicas

### 3. **Comparação de Modelos**
- Ative "Comparar todos os modelos" para análise comparativa
- Visualize métricas lado a lado
- Identifique o melhor algoritmo para seu dataset

## 🔍 Dataset: California Housing

O projeto utiliza o dataset **California Housing** que contém:

- **8 features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: MedHouseVal (valor mediano das casas em blocos censitários da Califórnia)
- **Tamanho**: 20,640 amostras
- **Tipo**: Regressão (valores contínuos)

## 🏗️ Arquitetura do Código

### Classe RandomForestModel (Implementação Profissional)
```python
class RandomForestModel:
    """
    Implementação profissional do Random Forest com:
    - Tratamento robusto de erros
    - Logging profissional
    - Métodos modulares e reutilizáveis
    - Documentação completa
    """
    
    def train_baseline_model(self, X_train, y_train)
    def evaluate_model(self, model, X_test, y_test)
    def optimize_hyperparameters_grid(self, X_train, y_train)
    def optimize_hyperparameters_random(self, X_train, y_train)
    def plot_feature_importance(self)
    def plot_n_estimators_analysis(self)
    def plot_prediction_scatter(self)
```

### Funções Principais
- **`run_random_forest_analysis()`**: Nova função principal profissional
- **`rodar_random_forest()`**: Função legacy para compatibilidade

## 📊 Métricas de Avaliação

### Regressão
- **MAE (Mean Absolute Error)**: Erro absoluto médio
- **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio
- **R² (Coefficient of Determination)**: Coeficiente de determinação

### Interpretação
- **MAE/RMSE**: Quanto menor, melhor (0 = perfeito)
- **R²**: Quanto mais próximo de 1, melhor (1 = perfeito)

## 🔧 Configurações e Personalização

### Parâmetros Configuráveis
```python
# Random Forest
RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
CV_FOLDS = 5
RANDOM_SEARCH_ITERATIONS = 20

# Hiperparâmetros de busca
GRID_SEARCH_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## 🚀 Próximos Passos

### Melhorias Planejadas
- [ ] Implementação profissional para todos os algoritmos
- [ ] Adição de mais datasets de exemplo
- [ ] Sistema de experimentos com MLflow
- [ ] API REST com FastAPI
- [ ] Deploy automatizado com CI/CD
- [ ] Testes unitários e de integração
- [ ] Documentação automática com Sphinx

### Novos Algoritmos
- [ ] Deep Learning com TensorFlow/PyTorch
- [ ] Ensemble methods avançados
- [ ] AutoML com Auto-sklearn
- [ ] Interpretabilidade com SHAP/LIME

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

### Padrões de Código
- Siga **PEP 8** para estilo Python
- Use **type hints** em todas as funções
- Documente com **docstrings** completas
- Implemente **tratamento de erros** robusto
- Adicione **logging** apropriado

## 📚 Recursos de Aprendizado

### Machine Learning
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Towards Data Science](https://towardsdatascience.com/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Boas Práticas
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Real Python](https://realpython.com/)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**João Silva** - [GitHub](https://github.com/seu-usuario)

## 🙏 Agradecimentos

- Dataset: [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- Comunidade scikit-learn
- Streamlit team
- Contribuidores open source

---

⭐ **Se este projeto foi útil, considere dar uma estrela!**

📧 **Contato**: [seu-email@exemplo.com](mailto:seu-email@exemplo.com)

🔗 **LinkedIn**: [Seu Perfil](https://linkedin.com/in/seu-perfil)
