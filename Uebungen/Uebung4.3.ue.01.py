import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)



ax = plt.axes(facecolor = '#E6E6E6')

plt.grid(color = 'white', linestyle = 'solid')
for spine in ax.spines.values():
    spine.set_visible(False)

plt.hist(data, edgecolor = 'blue' , color = 'red')
plt.show()

print(plt.style.available)

plt.style.use('ggplot')

plt.hist(data)
plt.show()
