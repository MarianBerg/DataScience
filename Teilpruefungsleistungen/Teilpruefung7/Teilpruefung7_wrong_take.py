#Aufgabe
#
#Nehmen wir das Neural Network, welches wir bereits programmiert haben, 
#und fügen in unserer NN-Klasse verschiedene Aktivierungsfunktionen zwischen den Schichten ein, 
#um zu untersuchen, wie sie die Performance beeinflussen. 
#
#    Recherchiere, wie du Sigmoid, Leaky ReLU, PReLU und ELU in PyTorch implementieren kannst.
#    Trainiere das Neural Network mit der Sigmoid-Funktion und halte die Performance fest.
#    Trainiere das Neural Network mit der Leaky-ReLU-Funktion für die Parameter 0.01, 0.05, 0.1, 0.5 und halte die Performance fest.
#    Trainiere das Neural Network mit der PreLU-Funktion.
#    Trainiere das Neural Network mit der ELU-Funktion für die Parameter 0.1, 0.2 und 0.3 und halte die Performance fest.


import torch.nn as nn
import torch
import pandas as pd
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
            nn.Linear(28*28, anzahl_neuronen_eingabeschicht),                           #zero layer
            nn.Sigmoid(),                                                               #first_layer
            nn.Linear(anzahl_neuronen_eingabeschicht, anzahl_neuronen_ausgabeschicht),  #second_layer
            nn.Sigmoid(),                                                               #third_layer    
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.network(x)
        return output
    
    def train(self, input_list, w_zero_layer, w_second_layer, target_list, learning_rate):
        input_vector = np.array(input_list, ndmin = 2).T

        #output der input Schicht berechnen
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        target_tensor = torch.tensor(target_list, dtype=torch.float32)
        
        zero_layer_output = self.network[0] (input_tensor)

        #erste sigmoid auf diesen output anwenden
        first_layer_output = self.network[1] (zero_layer_output)
        

        #output der zweiten Schicht berechnen
        second_layer_output = self.network[2](first_layer_output)

        #dritte schicht sigmoid anwenden 
        third_layer_output = self.network[3](second_layer_output)

        #zielwert berechnen
        

        #Fehler berechnen letzte Schicht im Original war das output_errors = targets - third_layer_output, 
        #das macht soweit ich verstehe aber
        #keinen Sinn, da die Idee des Backpropagation Algorithmus ist ein Minimum zu finden, diese Funktion aber kein Minimum hat.
        output_errors = ((target_tensor - third_layer_output)^2) / 2

        #Im naechsten Schritt muss fuer die Backpropagation vom Fehler zu den Parametern der zweiten Schicht folgende Ableitung berechnet werden:
        # d(output_errors)/d(w_second_layer) = 
        # (d(output_errors)/d(third_layer_output))      == -1*(targets - third_layer_output)
        # * (d(third_layer_output)/d(third_layer))      == 1 
        # * (d(third_layer) / d(second_layer_output))   == second_layer_output * (1 - second_layer_output)
        # * (d(second_layer_output) / d(w_second_layer))== first_layer_output

        
        backprop_from_error_to_second_layer = -1*(target_tensor - third_layer_output) * (second_layer_output * (1 - second_layer_output))
        
        w_second_layer -= learning_rate * np.dot(backprop_from_error_to_second_layer, np.transpose(first_layer_output) )

        #Nun muss die Backpropagation vom Fehler zu den Parametern der input oder zero Schicht folgende Ableitung durchgefuehrt werden:
        # d(output_errors)/d(w_first_layer) = 
        # (d(output_errors)/d(third_layer_output))           == -1*(targets - third_layer_output)
        # * (d(third_layer_output)/d(third_layer))           == 1 
        # * (d(third_layer) / d(second_layer_output))        == second_layer_output * (1 - second_layer_output) Kann vom ersten Teil widerverwendet werden.
        # * (d(second_layer_output) / d(first_layer_output)) == np.transpose(w_second_layer)
        # * (d(first_layer_output) / d(zero_layer_output))   == zero_layer_output * (1 - zero_layer_output)
        # * (d(zero_layer_output) / d(w_first_layer))        == np.transpose(input_vector)

        backprop_to_zero_layer  = np.dot(zero_layer_output * zero_layer_output * (1 - zero_layer_output), np.transpose(w_second_layer)) *  backprop_from_error_to_second_layer
        w_zero_layer -= learning_rate * np.dot(backprop_to_zero_layer, np.transpose(input_vector))


    def performance(self, data, labels):  

        richtige = []
        for input, label in  zip(data, labels):  
            output = self.forward(input)
            output = output.argmax()   
            if label == output:
                richtige.append(1) 
            else:
                richtige.append(0)

        test_size = data.__sizeof__
        performance = sum(richtige)/test_size
        print(test_size)
        print(sum(richtige))
        print(performance) 
        #numpy_dot( input_hidden*(1-input_hidden) * w_ho * output_errors * output_final*(1-output_final), numpy.transpose(input_vector  

def main():
    data = pd.read_csv('K4.0026_1.0_3.5.Ü.01_mnist_data.csv')

 

    labels = data.iloc[: , 0]
    label_list = labels.tolist()
    training_labels = label_list[:8000]

    

    images = data.iloc[: , 1:]
    images_list = images.values.tolist()
    training_images   = images_list[:8000]

    test_labels = label_list[8000:]
    test_images = images_list[8000: ]

    #skalierung:
    

    training_data   = (np.asfarray(training_images) / 255 * 0.99) + 0.01
    test_data       = (np.asfarray(test_images) / 255 * 0.99) + 0.01

    image = training_data[0]
    print(image.shape)
    neural_net = NN(10, 10)
    print(neural_net.forward(torch.tensor(training_data[2], dtype=torch.float32).unsqueeze(0)))

    #print(neural_net.performance(test_data, test_labels))

if __name__ == "__main__":
    main()