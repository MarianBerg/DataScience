from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('Mall_Customers.xls')

#Diesen Datensatz wollen wir mit Hilfe des K-Means-Algorithmus in Cluster einteilen. 
#Hierbei werden wir uns im ersten Schritt auf die zwei Features Annual Income und Spending Score konzentrieren. 
#Anschließend führen wir das Clustering für drei Features durch, wobei wir das Alter als drittes Feature hinzunehmen.
#
#Aufgabe 1:
#
#    Importiere alle wichtigen Bibliotheken.
#    Importiere die Daten und bereite sie vor.
#    Visualisiere die Daten in einem Streudiagramm.
#
color_function = {'Male': 'blue', 'Female': 'pink'}
colors = data['Gender'].map(color_function)

figure = plt.figure()
ax = figure.add_subplot(111, projection = '3d')

scatter_plot = ax.scatter(data['Annual Income (k$)'], data['Age'], data['Spending Score (1-100)'], c = colors)
ax.set_xlim([0, max(data['Annual Income (k$)']) + 1])
ax.set_ylim([0, max(data['Age']) + 1])
ax.set_zlim([0, max(data['Spending Score (1-100)']) + 1])

ax.set_xlabel('Annual Income')
ax.set_ylabel('Alter')
ax.set_zlabel('Spending Score')
plt.show()
plt.clf()


#Aufgabe 2:
#
#Bestimme die richtige Anzahl an Clustern.
#
#     Nutze die Elbow-Methode und plotte die Inertia-Werte.
#    Bestimme den Knick-Punkt.
#    Bestimme den Silhouetten-Koeffizient.
#

inertias = []
number_of_centroids = []
silhouetten = []

prepared_data = list(zip(data['Annual Income (k$)'], data['Spending Score (1-100)']))
for i in range(2, 12):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(prepared_data)

    inertias.append(kmeans.inertia_)
    number_of_centroids.append(i)
    labels = kmeans.labels_
    # Calculate silhouette score
    sil_score = silhouette_score(prepared_data, labels)    
    silhouetten.append(sil_score)


plt.scatter(number_of_centroids, inertias)
plt.xlabel('number_of_centroids')
plt.ylabel('inertia')
plt.show()
plt.clf()

plt.scatter(number_of_centroids, silhouetten)
plt.xlabel('number_of_centroids')
plt.ylabel('silhouette')
plt.show()
plt.clf()

#cluster == 5 is best

#Aufgabe 3:
#Lasse den K-Means-Algorithmus mit dem in Aufgabe 2 bestimmten K-Wert auf den Daten laufen und stelle die unterschiedlichen Cluster durch Farben getrennt in einem Streudiagramm dar.
#
kmeans = KMeans(n_clusters = 5)


kmeans.fit(prepared_data)
labels = kmeans.labels_
# Scatter plot with colors based on cluster labels


plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c = labels)
plt.show()
plt.clf

#Aufgabe 4:
#Jetzt führen wir das Clustering für drei Features durch und nehmen noch das Alter als Feature hinzu. 
#Stelle die Cluster in einem 3D-Koordinatensystem durch Farben getrennt dar.

prepared_data_3d = list(zip(data['Annual Income (k$)'], data['Age'], data['Spending Score (1-100)']))

kmeans3d = KMeans(n_clusters = 5)
kmeans3d.fit(prepared_data_3d)
labels3d = kmeans3d.labels_

figure = plt.figure()
ax = figure.add_subplot(111, projection = '3d')

scatter_plot = ax.scatter(data['Annual Income (k$)'], data['Age'], data['Spending Score (1-100)'], c = labels3d)
ax.set_xlim([0, max(data['Annual Income (k$)']) + 1])
ax.set_ylim([0, max(data['Age']) + 1])
ax.set_zlim([0, max(data['Spending Score (1-100)']) + 1])

ax.set_xlabel('Annual Income')
ax.set_ylabel('Alter')
ax.set_zlabel('Spending Score')
plt.show()
plt.clf()
