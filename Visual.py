import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

dataset = datasets.load_iris()
dataframe = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
dataframe['target'] = dataset.target
matrix = dataframe.corr()

# Heatmap of correlation matrix created using matplotlib
plt.imshow(matrix, cmap='Blues')
plt.colorbar()
variables = []
for i in matrix.columns:
    variables.append(i)

plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
plt.yticks(range(len(matrix)), variables)
plt.show()

# Heatmap of correlation matrix created using seaborn
sns.heatmap(matrix, cmap='Greens', annot=True)
