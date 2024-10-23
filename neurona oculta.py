import numpy as np

# Definir la función de activación sigmoide
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

# Definir los patrones de entrada
P1 = np.array([1, 0, 1])
P2 = np.array([1, 1, 0])

# Definir los vectores de pesos iniciales
W3 = np.array([1.5, 1, 1])
W4 = np.array([1, 1, -2])

# Definir el factor de aprendizaje
alpha = 0.1

# Realizar una iteración para el patrón P1
# Calcular la entrada neta de la neurona oculta
net_h = np.dot(W3, P1)

# Calcular la salida de la neurona oculta
out_h = sigmoid(net_h)

# Calcular la entrada neta de la neurona de salida
net_o = np.dot(W4, np.array([1, out_h, 1]))

# Calcular la salida de la neurona de salida
out_o = sigmoid(net_o)

# Calcular el error de salida para el patrón P1
error_o = P1[2] - out_o

# Calcular el error de la neurona oculta utilizando la retropropagación del error
error_h = out_h * (1 - out_h) * error_o * W4[1]

# Actualizar los pesos de la neurona oculta utilizando el algoritmo de Backpropagation
W3[1] += alpha * error_h * P1[0]
W3[2] += alpha * error_h * P1[1]

# Realizar una iteración para el patrón P2
# Calcular la entrada neta de la neurona oculta
net_h = np.dot(W3, P2)

# Calcular la salida de la neurona oculta
out_h = sigmoid(net_h)

# Calcular la entrada neta de la neurona de salida
net_o = np.dot(W4, np.array([1, out_h, 1]))

# Calcular la salida de la neurona de salida
out_o = sigmoid(net_o)

# Calcular el error de salida para el patrón P2
error_o = P2[2] - out_o

# Calcular el error de la neurona oculta utilizando la retropropagación del error
error_h = out_h * (1 - out_h) * error_o * W4[1]

# Actualizar los pesos de la neurona oculta utilizando el algoritmo de Backpropagation
W3[1] += alpha * error_h * P2[0]
W3[2] += alpha * error_h * P2[1]

# Imprimir los vectores de pesos actualizados
print("W3 = ", W3)
print("W4 = ", W4)