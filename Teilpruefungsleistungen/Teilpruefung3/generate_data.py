import pandas as pd
import numpy as np

input = np.random.rand(200) * 10

output = -0.4* input**4 + 4*input **2 + 2 * input + 6

data = pd.DataFrame({'input': input, 'output': output})

data.to_excel('test_data.xlsx', index = False)