import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy.stats import pearsonr
import numpy as np

import pandas as pd

data = pd.read_csv("K4.0026_1.0_2.C.01_Salaries.csv")

data = data.dropna()

x = data["Level"]

y = data["Salary"]

positions = data["Position"]
classes = data["Position"].unique()

correlation_coefficient = pearsonr(x, y)
print(correlation_coefficient)
quadratic, slope, intercept = np.polyfit(x, y, deg = 2)
regression = quadratic * x **2 + slope * x + intercept

number_of_cathegories = len(classes)
color_map = plt.cm.get_cmap('tab10', number_of_cathegories)



color_mapping = {category: color_map(i) for i, category in enumerate(classes)}

# Create a scatter plot with colors based on the 'Category' column
plt.scatter(x, y, c = positions.map(color_mapping), label=positions)

plt.plot(x, regression, color='black', linestyle='--', label='Linear Regression')


# Add labels and legend
plt.xlabel('Level')
plt.ylabel('Salary')



# Create legend with colors and meanings
legend_handles = [Patch(color=color_mapping[category], label=category) for category in classes]
plt.legend(handles=legend_handles, title='Categories')


# Show the plot
plt.show()

