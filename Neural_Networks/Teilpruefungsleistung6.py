import pandas as pd
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt 

import time

import torch.nn as nn
import torch.optim as optim
import torch

#Aufgabe
#
#Aufsetzen des Neural Networks
#
#Einige Grundparameter haben wir ja bereits festgelegt. Wir werden mit einem drei-schichtigen Neural Network arbeiten. 
#Da die Daten 784 Pixel haben, wird die Eingangsschicht 784 Neuronen bekommen. 
#Und da wir Zahlen zwischen 0 und 9 voneinander unterscheiden wollen, wird die Ausgabeschicht 10 Neuronen bekommen. 
#Die Anzahl der Neuronen im Hidden Layer setzen wir willkürlich auf 100. Ebenso willkürlich setzen wir die Lernrate auf 0.3. 
#Wie sich diese Werte auf die Performance des Neural Networks auswirken, werden wir später untersuchen. 
#
#    Erzeuge die Gewichts-Matrizen durch zufälliges Setzen von Werten zwischen -0.5 und 0.5. Verwende hierfür das numpy.random-Modul
#    Schreibe eine Funktion zur Signalübertragung im Neural Network. 
#Da wir sie später zum Testen des trainierten Neural Networks einsetzen werden, nennen wir die Funktion test(). 
#Als Argumente soll die Funktion den Input-Vektor und die beiden Gewichtsmatrizen enthalten.Zurückgeben soll die Funktion den Output-Vektor (ein 10-zeiliger Vektor).
#    Lasse dir mit print() den Output-Vektor vom ersten Element des Datensatzes ausgeben.

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        flatted_input = self.flatten(x)

        logits = self.network(flatted_input)
        return logits
    
    def test(self, input, input_weights, output_weights):
        # Manually set the layer parameters
        with torch.no_grad():
            self.network[0].weight = nn.Parameter(input_weights)

            self.network[2].weight = nn.Parameter(output_weights)

        
        # Forward pass with the provided input tensor
        output = self.forward(input)
        return output




def main():
    data = pd.read_csv('K4.0026_1.0_3.5.Ü.01_mnist_data.csv')

    #Daten vorbereiten
    converted_data = data.values.tolist()



    pixel_data_formated = []
    label_data = []
    for element in converted_data:

        label_data.append(np.asfarray(element[0]))
        pixel_data_formated.append( np.asfarray(element[1:]).reshape((28, 28)))

    #Daten splitten noch Teil von der Uebung
    train_data_label = label_data[:8000]
    train_data_input = pixel_data_formated[:8000]


    train_data_input_normalized = np.asfarray(train_data_input) * 0.99 / 256 + 0.01


    test_data_label = label_data[8000:]
    test_data_input = pixel_data_formated[8000:]

    test_data_input_normalized = np.asfarray(test_data_input) * 0.99 / 256 + 0.01

    #Hyperparameter hier noch nicht benutzt
    
    input_size = 784
    output_size = 10
    #learning_rate = 0.0001
    #num_epochs = 5
    model = NN()
    
    input_weights = torch.tensor(np.random.uniform(-0.5, 0.5, (100, 784)), dtype=torch.float32)
    output_weights = torch.tensor(np.random.uniform(-0.5, 0.5, (10, 100)), dtype=torch.float32)

    

    tensor_input = torch.tensor(train_data_input_normalized[0], dtype=torch.float32)

    #adding batch dimension
    tensor_input_batch = tensor_input.unsqueeze(0)
    
    output = model.test(tensor_input_batch, input_weights = input_weights, output_weights = output_weights)
    
    print("output: {} \n Shape: {}".format(output, output.shape))



if __name__ == "__main__":
    main()



