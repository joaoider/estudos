1. Fundamentos de Machine Learning
Tipos de aprendizado: supervisionado, não supervisionado, semi-supervisionado, por reforço

Problemas clássicos: classificação, regressão, clustering, redução de dimensionalidade
Modelos tradicionais:
Regressão linear e logística
KNN (K-Nearest Neighbors)
SVM (Support Vector Machines)
Árvores de decisão, Random Forests e Gradient Boosting (XGBoost, LightGBM, CatBoost)
Naive Bayes

2. Pré-processamento de Dados
Limpeza e tratamento de dados faltantes
Normalização vs padronização (MinMaxScaler, StandardScaler)
Codificação de variáveis categóricas (One-Hot, LabelEncoder)
Seleção de atributos (feature selection) e extração de atributos (feature engineering)
Redução de dimensionalidade (PCA, t-SNE, UMAP)
Detecção e tratamento de outliers

3. Divisão de Dados
Split em treino, validação e teste (hold-out, K-Fold, Stratified K-Fold)
Cross-validation e validação cruzada aninhada
Data leakage: como evitar

4. Redes Neurais Artificiais (ANN)
Estrutura de um neurônio artificial (modelo MCP, perceptron)
Arquitetura: camadas de entrada, ocultas e saída
Funções de ativação:
ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Swish
Inicialização de pesos (Xavier, He)
Seleção de número de camadas e neurônios

Otimização:

Algoritmos: SGD, Adam, RMSProp, Adagrad
Aprendizado por mini-batch
Learning rate, decay, momentum

5. Função de Custo e de Perda
Para regressão: MSE, MAE, Huber
Para classificação: Cross-Entropy, Log Loss
Custom Loss Functions
Função de avaliação (métrica) vs função de perda

6. Técnicas de Regularização
L1 (Lasso), L2 (Ridge)
Dropout
Early stopping
Batch normalization

7. Modelos Avançados de Redes Neurais
Redes Neurais Convolucionais (CNNs) – visão computacional
Redes Neurais Recorrentes (RNNs, LSTM, GRU) – séries temporais e NLP
Transformers e Attention Mechanisms – NLP moderno
Autoencoders e GANs – geração de dados e compressão
Transfer Learning
Redes profundas (DNN) e arquiteturas híbridas

8. Avaliação de Modelos
Métricas para classificação: Accuracy, Precision, Recall, F1, AUC-ROC
Métricas para regressão: RMSE, MAE, R²
Matriz de confusão
Curva ROC e PR
Bias-variance tradeoff

9. Ajuste e Seleção de Hiperparâmetros
Grid Search, Random Search
Bayesian Optimization
Optuna, Hyperopt, Keras Tuner

10. Produção e Deploy
Serialização de modelos (pickle, joblib, ONNX)
Deploy com Flask, FastAPI, Streamlit
Monitoramento de modelos em produção
MLOps: versionamento de modelos, pipelines, MLflow, Airflow

11. Casos Práticos e Projetos
Classificação de imagens com CNN
Previsão de séries temporais com LSTM
Análise de sentimentos com Transformers
Anomalia com Autoencoders
Recomendação com embeddings



LightGBM
CatBoost
SVR
Regressão Lasso / Ridge / ElasticNet
KNN 
HistGradientBoostingRegressor