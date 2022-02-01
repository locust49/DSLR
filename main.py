from re import L
from statistics import mode
from numpy import percentile
from sklearn.metrics import accuracy_score
from data_class import Data
from logistic_regression_class import LogisticRegression
import pandas as pd
import numpy as np
newData = Data.from_csv('./datasets/dataset_train.csv')

# pd.set_option("display.max_rows", None, "display.max_columns", None)
# Testing each method
# print('Count:\n' , newData.ft_count())
# print('Min:\n' , newData.ft_min())
# print('Max:\n' , newData.ft_max())
# print('Mean:\n' , newData.ft_mean())
# print('Std:\n' , newData.ft_std())
# print('perc:\n' , newData.ft_percentiles([25, 50, 75]))

# Testing the describe method
my_desc = newData.ft_describe(percentiles=[.25, .50, .75])
desc = newData.data.describe()

# Testing the describe method
# print('my describe:\n' , my_desc)
# print('em describe:\n' , desc)
# print('accuracy:\n', desc - my_desc)

model = LogisticRegression()


################################ TRAIN ONEVSALL ################################
df_train = pd.read_csv('./datasets/dataset_train.csv', index_col=0)
fill = df_train.fillna(df_train.mean())

data_x = fill.select_dtypes(np.number)
df_norm = data_x / data_x.max()
target_y = fill['Hogwarts House']

print(df_norm)
houses = ['Hufflepuff', 'Ravenclaw', 'Slytherin', 'Gryffindor']
# w = model.onevsall(df_norm, target_y, houses)



################################# TRAIN BINARY ################################
# df_train = pd.read_csv('./datasets/dataset_train.csv', index_col=0)
# fill = df_train.fillna(df_train.mean())

# data_x = fill.select_dtypes(np.number)

# target_y = fill[['Hogwarts House']]
# target_y.loc[target_y['Hogwarts House'] == 'Gryffindor', 'Hogwarts House'] = 1
# target_y.loc[target_y['Hogwarts House'] != 'Gryffindor', 'Hogwarts House'] = 0
# print(target_y)


# model.train(data_x, target_y)


################################ PREDICT ################################



df_predict = pd.read_csv('./datasets/dataset_test.csv', index_col=0)
df_predict.dropna(axis=1, how='all', inplace=True)
fill = df_predict.fillna(df_train.mean())
data_x = fill.select_dtypes(np.number)
df_norm = data_x / data_x.max()


# pred = model.predict(df_norm, w)

##################################### TEST #####################################

df_test = pd.read_csv('./datasets/dataset_truth.csv', index_col=0)
pred = pd.read_csv('./debug/my_pred.csv', index_col=0)

df_test['pred'] = pred

df_test['res'] = 0

df_test['res'] = np.where(df_test['pred'] == df_test['Hogwarts House'] , 1, 0)
print(df_test[df_test['res'] == 0])
print('accuracy : ', df_test['res'].mean())