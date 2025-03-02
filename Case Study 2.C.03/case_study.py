import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#Regressions-Projekt
#
#In diesem Projekt bekommst du Universitätsdaten, die in Cluster eingeteilt werden sollen. 
#Die interessanten Features hierbei sind F.Undergrad (Anzahl der Vollzeit-Undergraduate-Studierenden), 
#Applications (Anzahl der Bewerbungen), Accept (Anzahl der akzeptierten Bewerbungen), 
#Enroll (Anzahl der Bewerber*innen, die sich tatsächlich einschreiben) und Grad.Rate (die Rate der Studierenden, die einen Abschluss machen). 
#
#1.	Visualisiere die Daten. Stelle in einem Scatter-Plot folgende Zusammenhänge dar: 
#
#        Bewerbungen und akzeptierte Bewerbungen 
#        Bewerbungen und Anzahl neuer Studierenden 
#        Bewerbungen und Abschlussrate 
#
#2.	Wende nun einen Clustering-Algorithmus an, um in den obigen Plots Cluster zu identifizieren. 
#
#Achte darauf, die richtige Anzahl an Clustern zu bestimmen (Elbow-Methode, Silhouetten-Koeffizient).

data = pd.read_csv('K4.0026_1.0_2.C.03_UniData.csv')


Applications = data['Applications']
Accepted = data['Accept']



#plt.scatter(Applications, Accepted)
#plt.show()
#plt.clf()

from sklearn.cluster import KMeans

data_processing = list(zip(Applications, Accepted))

inertias = []
number_of_centroids = []
silhouetten = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters= i, n_init = 10, random_state=42)
    kmeans.fit(data_processing)

    inertias.append(kmeans.inertia_)
    number_of_centroids.append(i)
    labels = kmeans.labels_
    # Calculate silhouette score
    sil_score = silhouette_score(data_processing, labels)    
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


#plt.scatter(Applications, data['Grad_Rate'])
#plt.show()
#plt.clf()


