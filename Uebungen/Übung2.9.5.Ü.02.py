#Recherchiere, wie du dir von einem KMeans-Objekt die Koordinaten der Cluster-Zentren ausgeben lassen kannst. 
#Füge diese im Streudiagramm der Daten ein.

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

#creating random data with 5 centers, random_state = 2 is used to make it reproducible
data, _ = make_blobs(n_samples = 500, centers = 5, random_state = 2)



kmeans = KMeans(n_clusters = 5)
kmeans.fit(data)

centroids = kmeans.cluster_centers_


    

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 100)
plt.show()
