import torch.nn as nn
import torch
import numpy as np
import sklearn 

#Formuliere die Klasse NN() so, dass die Anzahl der Neuronen 
#in der Eingabeschicht und in der Ausgabeschicht als Anfangsparameter 
#der __init__()-Methode mitgegeben werden.

class NN(nn.Module):
    def __init__(self, anzahl_neuronen_eingabeschicht, anzahl_neuronen_ausgabeschicht):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, anzahl_neuronen_eingabeschicht),
            nn.Sigmoid(),
            nn.Linear(anzahl_neuronen_eingabeschicht, anzahl_neuronen_ausgabeschicht),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.network(x)
        return output
    
    def train(input_list, w_input_layer, w_output_layer, target_list, learning_rate):
        input_vector = np.array(input_list, ndmin = 2).T

        #output der input Schicht berechnen
        zero_layer_output = np.dot(w_input_layer, input_vector)

        #sigmoid auf diesen output anwenden
        first_layer_output = torch.sigmoid(zero_layer_output)

        #output der letzten Schicht berechnen
        third_layer_output = np.dot(w_output_layer, first_layer_output)

        #sigmoid anwenden 
        final_layer_output = torch.sigmoid(third_layer_output)

        #zielwert berechnen
        targets = np.array(target_list, ndmin = 2).T

        #Fehler berechnen letzte Schicht
        output_errors = targets - final_layer_output

        #Fehler in der Eingabeschicht
        zero_layer_error = np.dot(w_output_layer.T, output_errors)
        
        w_output_layer -= learning_rate * np.dot((output_errors * (final_layer_output * (1-final_layer_output)) ), np.transpose(first_layer_output))

        w_input_layer -= learning_rate * np.dot((zero_layer_error * first_layer_output * (1 - first_layer_output)), np.transpose(input_vector))
