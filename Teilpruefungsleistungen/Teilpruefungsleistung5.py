#Recherchiere, wie du mit der scipy-Bibliothek das Dendrogramm für die folgenden Daten ausgeben lassen kannst:

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

x = [4, 6, 9, 4, 3, 11, 12 , 6, 10, 12]

y = [22, 18, 25, 16, 16, 24, 24, 22, 21, 21]

data = list(zip(x, y))

#compute distance matrix
dendrogram_data = linkage(data, method='single')


#plot dendrogram
dendrogram(dendrogram_data)

plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Abstand')

plt.show()