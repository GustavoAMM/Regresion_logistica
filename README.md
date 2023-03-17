# Lógica

## Dentro del método fit(), se realiza una serie de pasos:

- Se calcula el número de elementos de la matriz de características X y se cambia el tamaño del vector de respuestas y para que tenga las mismas dimensiones que X.

- Si el booleano bias es True, se agrega una columna de unos a la matriz de características X para incluir la intercepción en el modelo.

- Se inicializan los valores de los coeficientes de theta a cero.

- Se crean dos listas vacías para almacenar los valores de pérdida y número de iteración.

- Se itera por un número de veces igual a epochs y se realiza el algoritmo de gradiente descendente para ajustar los valores de theta. En cada iteración, se calcula la hipótesis del modelo, el error y el gradiente, y se actualizan los valores de theta en consecuencia.

- Se almacenan los valores de pérdida y número de iteración en las listas creadas anteriormente.

- Finalmente, se devuelve una tupla con las dos listas.