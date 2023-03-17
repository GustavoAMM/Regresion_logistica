import numpy as np
import pandas as pd

class LogisticRegression():

    #Inicializar la clase. 
    #En este caso, no se hace nada, por lo que el método está vacío.
    def __init__(self) -> None:
        pass
    
    #Entrenar el modelo. 
    #El método utiliza el algoritmo de gradiente descendente para ajustar los valores de los coeficientes de theta a los datos de entrenamiento. 
    #Devuelve dos listas: 
    #   los valores de pérdida de cada iteración
    #   Número de iteración correspondiente.
    def fit(self, X, y, learning_rate=0.0001, epochs=1000, bias=True):
        n = int(len(X))  # numero de elementos de x
        y = np.resize(y, (n, 1))
        if bias:
            m = X.shape[1] + 1
            aux = np.ones((n, 1))
            X = np.concatenate((X, aux), axis=1)
        else:
            m = X.shape[1]
        thetas = np.zeros((m, 1)) #initial values of thetas

        errores = []
        iter_ = []

        for i in range(epochs):
            z = np.dot(X, thetas)
            h = self.h(z)
            error = h - y
            grad = np.dot(X.T, error) / n
            thetas -= learning_rate * grad
            iter_.append(i)
            loss = self.mean_squared_error(y, h)
            errores.append(loss)
        print(thetas)
        return (iter_, errores)

    #Calcula la hipótesis del modelo, que es la función sigmoidal.
    def h(self, z):
        return 1 / (1 + np.exp(-z))

    #Calcula la pérdida del modelo utilizando la función de pérdida de la regresión logística
    def mean_squared_error(self, y, h):
        n = len(y)
        loss = (-1 / n) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss
