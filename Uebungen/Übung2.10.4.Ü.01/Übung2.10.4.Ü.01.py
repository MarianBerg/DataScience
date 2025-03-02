from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Mall_Customers.xls')
prepared_data = list(zip(data['Annual Income (k$)'], data['Spending Score (1-100)']))


hierachycal_cluster = AgglomerativeClustering(n_clusters = 2, compute_distances=True)

hierachycal_cluster.fit(prepared_data)

distance_matrix  = hierachycal_cluster.distances_
print(distance_matrix)
