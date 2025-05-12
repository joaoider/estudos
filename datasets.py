# Medidas de flores iris (3 classes)
from sklearn.datasets import load_iris

# Imagens 8x8 de dígitos escritos à mão (0–9)
from sklearn.datasets import load_digits

# Preço de imóveis em Boston
from sklearn.datasets import load_boston
# Progressão da diabetes
from sklearn.datasets import load_diabetes	

# Qualidade de vinhos
from sklearn.datasets import load_wine		

# Diagnóstico de câncer
from sklearn.datasets import load_breast_cancer

# Preço de imóveis (atualizado do Boston)
from sklearn.datasets import fetch_california_housing	

# Cobertura do solo com 500k exemplos	
from sklearn.datasets import fetch_covtype		


import seaborn as sns
# Dados de sobrevivência no Titanic
sns.load_dataset('titanic') 		

# Gorjetas em restaurante
sns.load_dataset('tips')		

# Preço de diamantes por atributos
sns.load_dataset('diamonds')

# Mesmo do scikit-learn, versão tabular
sns.load_dataset('iris')	

# Passageiros por mês
sns.load_dataset('flights')	 	

# Dígitos manuscritos 28x28 (imagens em grayscale)
from tensorflow.keras.datasets import mnist		

# Roupas (camisetas, calças, etc.) em imagens
from tensorflow.keras.datasets import fashion_mnist		

# 10 categorias de imagens (avião, carro, etc.)
from tensorflow.keras.datasets import cifar10		

# Versão com 100 classes
from tensorflow.keras.datasets import cifar100	

# Sentimentos positivos/negativos em críticas
from tensorflow.keras.datasets import imdb	

# Classificação de tópicos em notícias
from tensorflow.keras.datasets import reuters	