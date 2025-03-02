import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#In diesem Projekt soll es darum gehen, den Preis für Mobiltelefone klassifizieren zu können. Hierbei ist für die Handys die Price Range in drei Klassen aufgeteilt: low, medium und high. Abhängig von Features wie der Batterieleistung, der Anzahl der Megapixel der Frontkamera, dem Speicher oder Bluetooth-Funktionen soll die Price Range eines Handys vorhergesagt werden. 

#1.	Visualisiere die folgenden Abhängigkeiten zwischen den Variablen in einem Scatter-Plot, 
#    wobei die Preisklasse mit Farben gekennzeichnet werden soll: 
#
#        Battery Power und Internal Memory 
#        Frontkamera-Megapixel und Bluetooth 
#        Internal Memory und Frontkamera-Megapixel 
#        3D-Scatter-Plot von Battery Power, Bluetooth und Frontkamera-Megapixel 



data = pd.read_csv("K4.0026_1.0_2.C.02_MobilePhone.csv")

plt.scatter(data['battery_power'], data['int_memory'])
plt.show()
plt.clf()


plt.scatter(data['frontcamermegapixels'], data['blue'])
plt.show()
plt.clf()


plt.scatter(data['int_memory'], data['frontcamermegapixels'])
plt.show()
plt.clf()

figure = plt.figure()
ax = figure.add_axes([0, 0, 1, 1], projection = '3d')
ax.scatter(data['battery_power'], data['blue'], data['frontcamermegapixels'])

ax.set_xlabel('Battery Power')
ax.set_ylabel('Bluetooth')
ax.set_zlabel('Frontkamera-Megapixels')

plt.show()
plt.clf()


#2.	Nutze einen Supervised-Classification-Algorithmus, um die Preisklasse eines Handys vorherzusagen.          
#        Features: Battery Power und Frontkamera-Megapixel 
#        Teile die Daten in Trainings- und Testdaten auf. 
#
#        Wähle einen geeigneten Algorithmus für eine Analyse. 
#        Trainiere den Algorithmus mit den Trainingsdaten. 
#        Teste den Algorithmus mit den Testdaten. 


mapping = {'l': 0, 'm': 1, 'h': 2}
classes = data['Price Range'].map(mapping)

import sklearn.ensemble as es
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

forest = es.RandomForestClassifier(n_estimators=100, criterion = 'entropy', bootstrap=False)

features = np.array([data['battery_power'], data['frontcamermegapixels']])
features_transposed = features.T

feature_train, feature_test, class_train, class_test = train_test_split(features_transposed, 
                                                                        classes, 
                                                                        test_size = 0.1, 
                                                                        random_state = 0,
                                                                    ) 
forest.fit(feature_train, class_train)

# Predict the target values for the test set
class_predict = forest.predict(feature_test)

# Evaluate the model
accuracy = accuracy_score(class_test, class_predict)
print("Accuracy:", accuracy)

# Detailed classification report
print("Classification Report:")
print(classification_report(class_test, class_predict))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(class_test, class_predict))