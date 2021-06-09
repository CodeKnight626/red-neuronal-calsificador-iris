# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:09:14 2021

@author: Cesar Fierro
"""


"""Importamos las librerias necesarias"""
import tensorflow as tf # Utilizaremos tensorflow, libreria dedicada la creacion de redes neuronales

import numpy as np #Utilizaremos tambien numpy para crear las matrices de entrada y guardar las matrices de salida


# Creamos los matrices de entrada y las rellenamos con numeros random
# Para cada flor tenemos una matriz de 50x4x1, tenemos 50 datos de cada tipo de flor, con 4 tipos de datos distintos
# Los llenamos con numeros random que actualizaremos mas adelante
iris_setosa_data = np.random.rand(50, 4, 1)
iris_versicolor_data = np.random.rand(50, 4, 1)
iris_virginica_data = np.random.rand(50, 4, 1)


"""Abrimos el archivo datos_iris.txt, en el que ajustaremos la informacion y la guardaremos para ajustarlo a la red neuronal"""
"""Utilizaremos el arcxhivo original sin modificaciones"""
with open("datos_iris.txt", encoding = 'utf-8') as fileData:
    
    raw_data = fileData.read() # Guardamos el texto en una variable que llamaremos raw_data
    
    # Iniciamos 3 ciclos "for" en el que guardaremos los datos completos para cada flor
    # Dividimos 3 ciclos de 50 iteraciones seguidos de otro ciclo con 4 iteraciones donde se guardaran lso 50 datos de los 3 tipos de flor
    #     ... Seguido de los 4 tipos de datos diferentes
    
    # Guardamos los datos de la flor setosa
    for i in range(0, 50):
        for j in range(0, 4):
                
            data = raw_data[:(raw_data.find(","))]
            iris_setosa_data[i][j][0] = float(data)
            raw_data = raw_data[raw_data.find(",") + 1:]
        raw_data = raw_data[raw_data.find("\n") + 1:]
    
    # Guardamos los datos de la flor versicolor
    for i in range(0, 50):
        for j in range(0, 4):
            data = raw_data[:(raw_data.find(","))]
            iris_versicolor_data[i][j][0] = float(data)
            raw_data = raw_data[raw_data.find(",") + 1:]
        raw_data = raw_data[raw_data.find("\n") + 1:]
        
    # Guardamos los datos de la flor virginica
    for i in range(0, 50):
        for j in range(0, 4):
            data = raw_data[:(raw_data.find(","))]
            iris_virginica_data[i][j][0] = float(data)
            raw_data = raw_data[raw_data.find(",") + 1:]
        raw_data = raw_data[raw_data.find("\n") + 1:]


"""Inicia el acomodo de datos y la separacion en datos de entrenamiento y prueba"""

# Una vez que tenemos los datos completos, hay que dividir la parte de entrenamiento y la parte de prueba

# Guardamos los primeros 30 valores de cada tipo de flor en 3 diferentes arreglos 
iris_setosa_data_train = iris_setosa_data[0:30]
iris_versicolor_data_train = iris_versicolor_data[0:30]
iris_virginica_data_train = iris_virginica_data[0:30]

"""Arreglo final de datos de entrada para la red"""
# Sumamos los 3 arreglos y en x_train guardamos el arreglo completo con los valores de entrenamiento
x_train = np.concatenate((iris_setosa_data_train, iris_versicolor_data_train, iris_virginica_data_train))


# Declaramos 3 arreglos distinos con ceros, de cardinalidad 30, los cuales llenaremos con el numero que representa cada flor 
iris_setosa_label_train = np.zeros(30) # Se declara el arreglo para entrenamiento con el tipo de la flor, de dimenson 30 que correspondera a los 30 datos de la flor setosa
iris_setosa_label_train.fill(1) # Se llena con unos y el numero uno correspondera a la flor setosa

iris_versicolor_label_train = np.zeros(30) # Se declara el arreglo para entrenamiento con el tipo de la flor, de dimenson 30 que correspondera a los 30 datos de la flor versicolor
iris_versicolor_label_train.fill(2) # Se llena con unos y el numero dos correspondera a la flor versicolor

iris_virginica_label_train = np.zeros(30)  # Se declara el arreglo para entrenamiento con el tipo de la flor, de dimenson 30 que correspondera a los 30 datos de la flor virginica
iris_virginica_label_train.fill(3) # Se llena con tres y el numero tres correspondera a la flor virginica

"""Arreglo final que contendra los valores esperados en cada indice parqa entrenamiento"""
# Se suman los 3 arreglos y tenemos el arreglo que contendra el tipo de la flor esperado, los primeros 30 indices flor setosa, del 31 al 60, la flor versicolor
#     ... y del indice 61 al 90 el tipo virginica
y_train = np.concatenate((iris_setosa_label_train, iris_versicolor_label_train, iris_virginica_label_train))


# Generamos 3 arreglos con los datos de los indices 30 al 50 para valores de prueba
iris_setosa_data_test = iris_setosa_data[30:50]
iris_versicolor_data_test = iris_versicolor_data[30:50]
iris_virginica_data_test = iris_virginica_data[30:50]

"""Arreglo final de datos de entrada para prueba"""
# Sumamos los 3 arreglos y obtenemos el arreglo con los valores de entrada de la red para la prueba
x_test = np.concatenate((iris_setosa_data_test, iris_versicolor_data_test, iris_virginica_data_test))

# Declaramos 3 arreglos de dimension 20 e igualmente que en el entrenamiento contendran lso valores esperados para cada flor
iris_setosa_label_test = np.zeros(20)
iris_setosa_label_test.fill(1)

iris_versicolor_label_test = np.zeros(20)
iris_versicolor_label_test.fill(2)

iris_virginica_label_test = np.zeros(20)
iris_virginica_label_test.fill(3)


"""Arreglo final que contendra los valores esperados en cada indice, utilizado para prueba"""

# Se suman los 3 arreglos de preuba con los valores esperados
y_test = np.concatenate((iris_setosa_label_test, iris_versicolor_label_test, iris_virginica_label_test))

"""Inicia creacion de la red neuronal"""

"""Utilizaremos keras de tensorflow para generar un modelo de red neuronal"""


# Definimos un modelo secuencial,de configuracion 4-32-4,
# la capa de salida tendra 4 neuronas y cada neurona correspondera a los 3 tipos de flor y un cuarto tipo de flor no encontrada
# Las funciones de activacion de la segunda capa sera la de relu y la de la capa de salida sera softmax
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(4,1)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
    ])


# Compilamos el modelo con eloptimizador adam, con una funcion de perdida sparse_categorical_crossentropy y decimos que nos muestre la precision
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# Pasamos los parametros de entenamiento y los ajustamos a 200 epocas
model.fit(x_train, y_train, epochs=200)

print("\nPerdida y precision: ")
# Evaluamos el modelo
model.evaluate(x_test, y_test, verbose=2)

# Hacemos una prediccion para todos los datos que tenemos para probar
# Este arreglo tendra una dimension de 4, 3 para cada tipo de flor y un cuarto para un tipo de flor no encontrado

predictions = model.predict(x_test)

# Declaramos un diccionario para intepretar los datos

tipo_de_flor = { 0 : "No se encontro tipo",
                 1 : "Iris Setosa",
                 2 : "Iris Versicolor",
                 3 : "Iris Virginica"}

print("\nPredicciones:\n")

# Recorremos el arreglo de predicciones e imprimimos el valor y el nombre de la flor que encontro
for prediction in predictions:
    max_value = np.where(prediction == np.amax(prediction))
    print(max_value[0][0], "\t", tipo_de_flor[max_value[0][0]])
