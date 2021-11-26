import numpy as np
import pandas as pd
import math

class Data:
	def __init__(self, data_file):
		print('Data Class constructor being called')
		temporary_data = pd.read_csv(data_file, index_col=0)
		self.data = temporary_data.select_dtypes(include=[np.number])
		self.columns = list(self.data.columns)
		self.number_of_columns = len(self.data.columns)
		self.summed_data = self.ft_sum()
		self.counted_data = self.ft_count()
		self.mean_data = self.ft_mean()

		print(self.data.describe())

	def ft_count(self):
		count = pd.DataFrame([[float(0)] * self.number_of_columns ], columns=list(self.columns))

		for index, observation in self.data.iterrows():
			for column in self.columns:
				# Calculate count
				if observation[column] is not None :
					if not np.isnan(observation[column]):
							count.loc[0][column] += 1
		return count

	def ft_min(self):
		minimum = self.data.loc[[0]]
		for index, observation in self.data.iterrows():
			for i in self.columns:
				# Calculate min
				if observation[i] <= minimum.loc[0][i]:
					minimum.loc[0, i] = observation[i]
		return minimum

	def ft_max(self):
		maximum = self.data.loc[[0]]
		for index, observation in self.data.iterrows():
			for i in self.columns:
				# Calculate max
				if observation[i] >= maximum.loc[0][i]:
					maximum.loc[0, i] = observation[i]
		return maximum

	def ft_sum(self):
		summed_data = pd.DataFrame([[float(0)] * self.number_of_columns ], columns=list(self.columns))

		for index, observation in self.data.iterrows():
			for column in self.columns:
				if observation[column] is not None:
					if not np.isnan(observation[column]):
						summed_data[column] += observation[column]
					else :
						summed_data[column] += 0
		return summed_data

	def ft_mean(self):
		mean = pd.DataFrame([[float(0)] * self.number_of_columns ], columns=list(self.columns))

		for i in self.columns:
			if self.counted_data.loc[0][i] == 0 :
				mean[i] = np.nan
			else:
				mean[i] = float(self.summed_data[i]) / float(self.counted_data.loc[0][i])
		return mean

	def	ft_percentiles(self, n_th):
		indexes = [id + '%' for id in list(map(str, n_th))]
		percentiles_data = pd.DataFrame(data=[[float(0)] * self.number_of_columns], index=indexes, columns=list(self.columns))

		# print(self.counted_data)
		# print(self.data.count())

		for n_percentile, id_percentile in zip(n_th, indexes):
			for id_column in range(self.number_of_columns):
				sorted_data = self.data.sort_values(by=[self.columns[id_column]])
				index = n_percentile * (int(self.counted_data.loc[0][id_column]) - 1) / 100
				# print('n_percent: [', n_percentile, '] INDEX INT = ', index.is_integer(), index, sorted_data.iloc[int(index)][id_column])
				if index.is_integer():
					percentiles_data.loc[id_percentile][id_column] = sorted_data.iloc[int(index)][id_column]
				else:
					percentiles_data.loc[id_percentile][id_column] = (sorted_data.iloc[math.ceil(index) - 1][id_column] + sorted_data.iloc[math.ceil(index)][id_column]) / 2
				# print(sorted_data)
		return percentiles_data


	def ft_std(self):
		std = pd.DataFrame([[float(0)] * self.number_of_columns ], columns=list(self.columns))

		squared_deviations_mean = [0] * self.number_of_columns
		for index, observation in self.data.iterrows():
			for id_column, column in zip(range(self.number_of_columns), self.columns):
				if not np.isnan(observation[column]):
					squared_deviations_mean[id_column] += (math.fabs(observation[column] - self.mean_data[column]))**2
		for id_column, column in zip(range(self.number_of_columns), self.columns):
			if self.counted_data.loc[0][id_column] != 1:
				std[column] = math.sqrt(squared_deviations_mean[id_column] / (self.counted_data.loc[0][id_column] - 1))
			else:
				std[column] = math.sqrt(squared_deviations_mean[id_column])
		return std

	def ft_describe(self, percentiles=None):
		indexes_percentiles = [id + '%' for id in list(map(str, percentiles))]

		rows_index = ['count',
					'mean',
					'std',
					'min',
					*indexes_percentiles,
					'max',
		]
		describe = pd.DataFrame(columns=self.columns, index=rows_index)
		describe.loc['count'] = self.ft_count().loc[0]
		describe.loc['mean'] = self.ft_mean().loc[0]
		describe.loc['std'] = self.ft_std().loc[0]
		describe.loc['min'] = self.ft_min().loc[0]
		percentiles_results = self.ft_percentiles(percentiles)
		for n_th in indexes_percentiles:
			describe.loc[n_th] = percentiles_results.loc[n_th]
		describe.loc['max'] = self.ft_max().loc[0]
		return describe