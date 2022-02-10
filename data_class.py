import	numpy	as np
from	numpy.lib	import percentile
from pandas.core.frame import DataFrame
from	tools		import *
import	pandas	as pd
import	math

class Data:
	def __init__(self, dataframe: DataFrame, normalize=False):
		'''
		Constructor of Data:
			Get the data from the dataset file
			Use the numerical data only
			Store this dataframe infos such as:
				'initial data'		: All the dataframe.
				'numerical_data'	: Only the numerical values of the dataframe.
				'columns'			: List of the columns' name dataframe.

				'summed_data'		: Dataframe of the sum of the numerical_data/column.
				'counted_data'		: Dataframe of the count of the numerical_data/column.
				'mean_data'			: Dataframe of the mean of the numerical_data/column.

				'self.normalized'	: If normalize is True, contains the normalized data of the
									  numerical_data.

		Args:
			normalize	: Default to False. If true, call the z_score_normalization method
						  and return a normalized version of the dataframe in self.normalized
		'''

		print('Data Class constructor has been called')


		self.initial_data = dataframe
		self.numerical_data = dataframe.select_dtypes(include=[np.number])

		self.columns = list(self.numerical_data.columns)

		self.summed_data = self.ft_sum()
		self.counted_data = self.ft_count()
		self.mean_data = self.ft_mean()

		if normalize is True:
			self.normalized = self.z_score_normalization()


	@classmethod
	def from_csv(cls, csv_filename, normalize=False):
		'''
			A 'classmethod' constructor that gets the data from a CSV file.
		'''
		try:
			data = pd.read_csv(csv_filename, index_col=0)
			return cls(data, normalize)
		except Exception:
			print('An error occured while trying to create a dataframe from csv.')

	def ft_count(self):
		'''
		count	method:
			counts the number of non-NA/null rows.
		'''
		count = { key:float(0) for key in self.columns }

		for index, row in self.numerical_data.iterrows():
			for column in self.columns:
				if row[column] is not None :
					if not np.isnan(row[column]):
							count[column] += 1
		return count

	def ft_min(self):
		'''
		min		method:
			returns the minimal value of the rows.
		'''
		minimum = self.numerical_data.iloc[0].to_dict()

		for index, row in self.numerical_data.iterrows():
			for column in self.columns:
				if row[column] <= minimum[column]:
					minimum[column] = row[column]
		return minimum

	def ft_max(self):
		'''
		max		method:
			returns the maximal value of the rows.
		'''
		maximum = self.numerical_data.iloc[0].to_dict()

		for index, row in self.numerical_data.iterrows():
			for column in self.columns:
				if row[column] >= maximum[column]:
					maximum[column] = row[column]
		return maximum

	def ft_sum(self):
		'''
		sum		method:
			returns the sum of the rows values.
		'''
		summed_data = { key:float(0) for key in self.columns }

		for index, row in self.numerical_data.iterrows():
			for column in self.columns:
				if row[column] is not None:
					if not np.isnan(row[column]):
						summed_data[column] += row[column]
					else :
						summed_data[column] += 0
		return summed_data

	def ft_mean(self):
		'''
		mean		method:
			returns the mean of the rows values.
			i.e:
				The arithmetic mean is the sum of the rows divided
				by their count (number of elements).
		'''
		mean = { key:float(0) for key in self.columns }

		for column in self.columns:
			if self.counted_data[column] == 0 :
				mean[column] = np.nan
			else:
				mean[column] = float(self.summed_data[column]) / float(self.counted_data[column])
		return mean

	def	ft_percentiles(self, n_th):
		'''
		percentiles	method:
			Returns the n-th percentile(s) of the rows.
			n_th:
				The default is [.25, .5, .75], which returns the 25th, 50th, and 75th percentiles.
				All should fall between 0 and 1.
		'''

		# Format the indexes (input : .25 | output : 25%)
		indexes = Tools.ft_format_percentiles(n_th)

		if indexes == None:
			return None

		percentiles_data = { index:{ key:float(0) for key in self.columns } for index in indexes}

		for n_percentile, id_percentile in zip(n_th, indexes):
			for id_column, column in enumerate(self.columns):
				# Sort the data by column
				sorted_data = self.numerical_data.sort_values(by=[self.columns[id_column]])
				# Compute the index of the percentile
				index = n_percentile * (int(self.counted_data[column]) - 1)

				if index.is_integer():
					percentiles_data[id_percentile][column] = sorted_data.iloc[int(index)][column]
				else:
					percentiles_data[id_percentile][column] = (
										sorted_data.iloc[math.ceil(index) - 1][column]
										+ sorted_data.iloc[math.ceil(index)][column]
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
		std = { key:float(0) for key in self.columns }

		squared_deviations_mean = { key:float(0) for key in self.columns }
		for index, row in self.numerical_data.iterrows():
			for column in self.columns:
				if not np.isnan(row[column]):
					squared_deviations_mean[column] += (math.fabs(row[column] - self.mean_data[column]))**2
		for column in self.columns:
			if self.counted_data[column] != 1:
				std[column] = math.sqrt(squared_deviations_mean[column] / (self.counted_data[column] - 1))
			else:
				std[column] = math.sqrt(squared_deviations_mean[column])
		return std

	def ft_describe(self, percentiles=None):
		'''
			Implementation of the DataFrame.describe method.
			Returns a dataframe of the 'count', 'mean', 'std',
				'min', 'percentiles', and 'max' methods
		'''

		indexes_percentiles = Tools.structure_percentiles(percentiles)
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

		describe.loc['count'] = self.ft_count()
		describe.loc['mean'] = self.ft_mean()
		describe.loc['std'] = self.ft_std()
		describe.loc['min'] = self.ft_min()
		percentiles_results = self.ft_percentiles(percentiles)
		for n_th in indexes_percentiles:
			describe.loc[n_th] = percentiles_results[n_th]
		describe.loc['max'] = self.ft_max()

		return describe