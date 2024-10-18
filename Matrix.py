import numpy as np
import pandas as pd
from sklearn import datasets

# 1: ice cream sales pattern by numpy

# x represents the total sale in dollars
x1 = [215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408]

# y represents the temperature on each day of sale
y1 = [14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2]

matrix1 = np.corrcoef(x1, y1)
print(matrix1)

# 2: glucose level in body pattern by numpy

# x represents the age
x2 = [43, 21, 25, 42, 57, 59]

# y represents the glucose level corresponding to that age
y2 = [99, 65, 79, 75, 87, 81]

matrix2 = np.corrcoef(x2, y2)
print(matrix2)

# 3: correlation matrix by pandas
data = {'x': [45, 37, 42, 35, 39], 'y': [38, 31, 26, 28, 33], 'z': [10, 15, 17, 21, 12]}
dataframe = pd.DataFrame(data, columns=['x', 'y', 'z'])
print('data frame is: \n', dataframe)

matrix = dataframe.corr()
print('\n correlation matrix is: \n', matrix)

# 4: correlation matrix by pandas on iris dataset
dataset = datasets.load_iris()
dataframe = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
dataframe['target'] = dataset.target

matrix = dataframe.corr()
print(matrix)
