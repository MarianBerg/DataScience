

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("KNN_Uebung.csv")

knn = KNeighborsClassifier(n_neighbors = 5)

x = data["Temperatur [°C]"]

y = data["Luftfeuchte [%]"]

bool_map = {'Ja': True, 'Nein': False}


classes = data['Draussen '].map(bool_map)

prepared_data = pd.concat([x, y], axis = 1)
knn.fit(prepared_data, classes)

new_x = 16
new_y = 60
new_point = [(new_x, new_y)]

prediction = knn.predict(new_point)

print(prediction)

plt.scatter(x, y, c = classes)

plt.scatter(new_x, new_y, color = 'green')

plt.show()