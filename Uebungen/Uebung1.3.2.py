import pandas as pd
import math


data = [32, 92, 4, 3, 99, 23, 1 ]

mue = sum(data) / len(data)

added_squares = 0
for i in range(0, len(data) - 1):
    added_squares += (mue - data[i]) * (mue - data[i])

sigma = math.sqrt(added_squares / len(data))

print(sigma)