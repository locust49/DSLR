import	numpy	as np
from	numpy.lib	import percentile
from	tools		import *
import	pandas	as pd
import	math

class Data:
	'''
	Constructor of Data:
		Get the data from the dataset file
		Use the numerical data only
		Store this dataframe infos such as:
			'columns' : List of the columns' name dataframe
			'number_of_columns' : Length of the 'columns' list
			'summed_data'		: Dataframe of the sum of the data/column
			'counted_data'		: Dataframe of the count of the data/column
			'mean_data'			: Dataframe of the mean of the data/column
	'''
	def __init__(self, data_file):
		print('Data Class constructor being called')
		temporary_data = pd.read_csv(data_file, index_col=0)
		self.data = temporary_data.select_dtypes(include=[np.number])
		self.columns = list(self.data.columns)
		self.number_of_columns = len(self.data.columns)
		self.summed_data = self.ft_sum()
		self.counted_data = self.ft_count()
		self.mean_data = self.ft_mean()


	def ft_count(self):
		'''
		count	method:
			counts the number of non-NA/null observations.
		'''
		count = pd.DataFrame([[float(0)] * self.number_of_columns ], columns=list(self.columns))

		for index, observation in self.data.iterrows():
			for column in self.columns:
				# Calculate count
				if observation[column] is not None :
					if not np.isnan(observation[column]):
							count.loc[0][column] += 1
		return count

	def ft_min(self):
		'''
		min		method:
			returns the minimal value of the observations.
		'''
		minimum = self.data.loc[[0]]
		for index, observation in self.data.iterrows():
			for i in self.columns:
				# Calculate min
				if observation[i] <= minimum.loc[0][i]:
					minimum.loc[0, i] = observation[i]
		return minimum

	def ft_max(self):
		'''
		max		method:
			returns the maximal value of the observations.
		'''
		maximum = self.data.loc[[0]]
		for index, observation in self.data.iterrows():
			for i in self.columns:
				# Calculate max
				if observation[i] >= maximum.loc[0][i]:
					maximum.loc[0, i] = observation[i]
		return maximum

	def ft_sum(self):
		'''
		sum		method:
			returns the sum of the observations values.
		'''
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
		'''
		mean		method:
			returns the mean of the observations values.
			i.e:
				The arithmetic mean is the sum of the observations divided
				by their count (number of elements).
		'''
		mean = pd.DataFrame([[float(0)] * self.number_of_columns ], columns=list(self.columns))

		for i in self.columns:
			if self.counted_data.loc[0][i] == 0 :
				mean[i] = np.nan
			else:
				mean[i] = float(self.summed_data[i]) / float(self.counted_data.loc[0][i])
		return mean




	def	ft_percentiles(self, n_th):
		'''
		percentiles	method:
			Returns the n-th percentile(s) of the observations.
			n_th:
				The default is [.25, .5, .75], which returns the 25th, 50th, and 75th percentiles.
				All should fall between 0 and 1.
		'''
		# Format the indexes (input : .25 | output : 25%)
		indexes = Tools.ft_format_percentiles(n_th)

		percentiles_data = pd.DataFrame(data=[[float(0)] * self.number_of_columns], index=indexes, columns=list(self.columns))

		for n_percentile, id_percentile in zip(n_th, indexes):
			for id_column in range(self.number_of_columns):
				# Sort the data by column
				sorted_data = self.data.sort_values(by=[self.columns[id_column]])
				# Compute the index of the percentile
				index = n_percentile * (int(self.counted_data.loc[0][id_column]) - 1)

				if index.is_integer():
					percentiles_data.loc[id_percentile][id_column] = sorted_data.iloc[int(index)][id_column]
				else:
					percentiles_data.loc[id_percentile][id_column] = (
										sorted_data.iloc[math.ceil(index) - 1][id_column]
										+ sorted_data.iloc[math.ceil(index)][id_column]
										) / 2
		return percentiles_data


	def ft_std(self):
		'''
		std		method:
			The `standard deviation` is the square root of the average of the
			squared deviations from the mean
			i.e.:
				`std = sqrt(mean(x))`, where `x = abs(obs - obs.mean())**2`.
		'''
		std = pd.DataFrame([[float(0)] * self.number_of_columns ],
							columns=list(self.columns))

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
		'''
			Implementation of the DataFrame.describe method.
			Returns a dataframe of the 'count', 'mean', 'std',
				'min', 'percentiles', and 'max' methods
		'''

		# Set default percentiles to [.25, .5, .75]
		if percentiles == None:
			percentiles = [.25, .5, .75]
		indexes_percentiles = Tools.ft_format_percentiles(percentiles)

		# If invalid data in percentiles (n < 0 or n > 1) return None
		if indexes_percentiles == None:
			return None

		# Initialize the result dataframe.
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