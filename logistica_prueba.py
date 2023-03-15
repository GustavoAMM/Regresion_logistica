#J = (-1/m) * sum(y * log(h) + (1-y) * log(1-h))

from sklearn import datasets
import logistic_regression as lr
from matplotlib import pyplot as plt
import numpy as np


data_set = datasets.load_breast_cancer()
X = data_set['data']
X.shape
y = data_set['target']
y.shape

# Corrige el error de que los valores de X estén muy alejados
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

lr = lr.LogisticRegression()
iteraciones, errores = lr.fit(X, y)

plt.plot(iteraciones, errores)
plt.show()
