import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from visualization_class import DataVisualization

df = pd.read_csv('./datasets/dataset_train.csv', index_col=0)
df_vis = DataVisualization(df, 'Hogwarts House')

print(df.describe())

df_vis.pairplot()
plt.show()
