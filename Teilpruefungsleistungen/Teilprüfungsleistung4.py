#Schreibe eine for-Schleife, die den K-Means-Algorithmus für 1, 2, 3, ..., 11 Cluster auf den Daten laufen lässt. 
#Recherchiere, wie du dir vom KMeans-Objekt den Inertia-Wert ausgeben lassen kannst, und speichere diesen in einer Liste. 
#Erzeuge dann einen Plot, in welchem du das Inertia auf der y-Achse gegen die Anzahl der Cluster auf der x-Achse aufträgst.

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

#creating random data with 5 centers, random_state = 2 is used to make it reproducible
data, _ = make_blobs(n_samples = 500, centers = 5, random_state = 2)

inertias = []
number_of_centroids = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data)

    inertias.append(kmeans.inertia_)
    number_of_centroids.append(i)

    

plt.scatter(number_of_centroids, inertias)
plt.show()



