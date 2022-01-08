from numpy import percentile
from data_class import Data

newData = Data.from_csv('./datasets/dataset_test.csv')


# Testing each method
# print('Count:\n' , newData.ft_count())
# print('Min:\n' , newData.ft_min())
# print('Max:\n' , newData.ft_max())
# print('Mean:\n' , newData.ft_mean())
# print('Std:\n' , newData.ft_std())
# print('perc:\n' , newData.ft_percentiles([25, 50, 75]))

# Testing the describe method
my_desc = newData.ft_describe(percentiles = [.89, .40, .6])
desc = newData.data.describe(percentiles = [.89, .40, .6])

# Testing the describe method
print('my describe:\n' , my_desc)
print('em describe:\n' , desc)
print('accuracy:\n', desc - my_desc)
