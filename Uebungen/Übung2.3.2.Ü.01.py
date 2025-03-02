import pandas as pd
from sklearn import linear_model

data = pd.read_csv("K4.0026_1.0_2.3.2.Ü.01_Fish.csv")

print(data)
X = data[["Length", "Height"]]
Y = data["Width"]


regression = linear_model.LinearRegression()


one_hot_species = pd.get_dummies(data["Species"])


independent_data = pd.concat([data[["Weight", "Length", "Height"]], one_hot_species], axis = 1)
print(independent_data)

regression.fit(independent_data, Y)


#Triff eine Vorhersage dazu, welche Breite ein Whitefish mit einem Gewicht von 233 Gramm, 
#einer Länge von 20.3 Zentimetern und einer Höhe von 13.32 Zentimetern hat.

prediction = regression.predict([[233, 20.3, 13.32, 0, 0, 0, 0, 0, 0, 1]])

print(prediction)