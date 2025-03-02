import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



#created the data with a 4 degree polynomial so 4 degree should be best, included degree 5 for comparison
data = pd.read_excel("test_data.xlsx")

X = data[['input']]
Y = data['output']


for i in range(1, 6):

    #encode the degree in the input, because that is the way it is appareantly done?!
    poly_features = PolynomialFeatures(degree = [i, i])
    x_poly = poly_features.fit_transform(X)

    #set up model
    model = LinearRegression()


    # Setup cross-validation
    k_fold = KFold(n_splits = 10, shuffle = True, random_state=0)

    # Perform cross-validation
    scores = cross_val_score(model, x_poly, Y, cv = k_fold, scoring = 'neg_mean_squared_error')

    # Output the results
    print("Polynomial_degree: {}".format(i))
    print("Cross-validated scores: {}".format(scores))
    print("Average Mean Squared Error: {}\n".format(-scores.mean()))

