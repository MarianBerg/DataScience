#Schreibe einen Python-Code, der die obige Signalübertragung vom Input-Vektor zum Output-Vektor berechnet. 
#Recherchiere hierfür, wie du mit der numpy-Bibliothek Matrizen darstellen und miteinander multiplizieren kannst. 
#Du kannst außerdem auch die Sigmoid-Funktion einfach mit dem scipy-Modul berechnen.

import numpy as np
import tensorflow as tf 

w_input_to_hidden = np.array([[0.9, 0.3, 0.4],
                              [0.2, 0.8, 0.2],
                              [0.1, 0.5, 0.6],
                              ])

w_hidden_to_output = np.array([[0.3, 0.7, 0.5],
                               [0.6, 0.5, 0.2],
                               [0.8, 0.1, 0.9],
                               ])

input_vector = np.array([0.9, 0.1, 0.8])


inbetween_result = np.dot(w_input_to_hidden, input_vector)

inbetween_output = tf.nn.sigmoid(inbetween_result)

result = tf.nn.sigmoid(np.dot(w_hidden_to_output, inbetween_output))

print(result)