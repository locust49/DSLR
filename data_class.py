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


    def __init__(self, data_file, default_count=False, default_sum=False, default_mean=False, default_min=False, default_percentiles=False, default_max=False):
        print('Data Class constructor being called')

        # Tracker attributes of methods call
        self.default_count = default_count
        self.default_sum = default_sum
        self.default_mean = default_mean
        self.default_min = default_min
        self.default_percentiles  = default_percentiles
        self.default_max = default_max

        # Change this to set it general as DataFrame !
        temporary_data = pd.read_csv(data_file, index_col=0)
        self.data = temporary_data.select_dtypes(include=[np.number])

        # Attributes to access redundant informations
        self.columns = list(self.data.columns)
        self.number_of_columns = len(self.data.columns)
        self.describe = self.init_describe()
        self.summed_data = self.ft_sum()
        self.counted_data = self.ft_count()
        self.mean_data = self.ft_mean()


    @classmethod
    def from_csv(cls, csv_filename):
        data = pd.read_csv(csv_filename, index_col=0)
        return cls(data)

    def ft_count(self):
        '''
        count	method:
            counts the number of non-NA/null observations.
        '''

        if self.default_count is True:
            return self.describe.loc['count']

        for index, observation in self.data.iterrows():
            for column in self.columns:
                # Calculate count
                if observation[column] is not None :
                    if not np.isnan(observation[column]):
                            self.describe.loc['count'][column] += 1
        self.default_count = True
        return self.describe.loc['count']

    def ft_min(self):
        '''
        min		method:
            returns the minimal value of the observations.
        '''

        if self.default_min is True:
            return self.describe.loc['min']
        for index, observation in self.data.iterrows():
            for i in self.columns:
                # Calculate min
                if observation[i] <= self.describe.loc['min'][i]:
                    self.describe.loc['min'][i] = observation[i]
        self.default_min = True
        return self.describe.loc['min']

    def ft_max(self):
        '''
        max		method:
            returns the maximal value of the observations.
        '''

        if self.default_max is True:
            return self.describe.loc['max']

        for index, observation in self.data.iterrows():
            for i in self.columns:
                # Calculate max
                if observation[i] >= self.describe.loc['max'][i]:
                    self.describe.loc['max'][i] = observation[i]
        self.default_max = True
        return self.describe.loc['max']

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

        if self.default_mean is True:
            return self.describe.loc['mean']
        for column in self.columns:
            if self.counted_data[column] == 0 :
                self.describe.loc['mean'][column] = np.nan
            else:
                self.describe.loc['mean'][column] = float(self.summed_data[column]) / float(self.counted_data[column])
        self.default_mean = True
        return self.describe.loc['mean']




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

        if indexes == None:
            return None

        if self.default_percentiles is True:
            return self.describe.loc[indexes]

        for n_percentile, id_percentile in zip(n_th, indexes):
            for id_column in range(self.number_of_columns):
                # Sort the data by column
                sorted_data = self.data.sort_values(by=[self.columns[id_column]])
                # Compute the index of the percentile
                index = n_percentile * (int(self.counted_data[id_column]) - 1)

                if index.is_integer():
                    self.describe.loc[id_percentile][id_column] = sorted_data.iloc[int(index)][id_column]
                else:
                    self.describe.loc[id_percentile][id_column] = (
                                        sorted_data.iloc[math.ceil(index) - 1][id_column] \
                                        + sorted_data.iloc[math.ceil(index)][id_column] \
                                        ) / 2
        self.default_percentiles = True
        return self.describe.loc[indexes]


    def ft_std(self):
        '''
        std		method:
            The `standard deviation` is the square root of the average of the
            squared deviations from the mean
            i.e.:
                `std = sqrt(mean(x))`, where `x = abs(obs - obs.mean())**2`.
        '''

        squared_deviations_mean = [0] * self.number_of_columns
        for index, observation in self.data.iterrows():
            for id_column, column in zip(range(self.number_of_columns), self.columns):
                if not np.isnan(observation[column]):
                    squared_deviations_mean[id_column] += (math.fabs(observation[column] - self.mean_data[column]))**2
        for id_column, column in zip(range(self.number_of_columns), self.columns):
            if self.counted_data[id_column] != 1:
                self.describe.loc['std'][column] = math.sqrt(squared_deviations_mean[id_column] / (self.counted_data[id_column] - 1))
            else:
                self.describe.loc['std'][column] = math.sqrt(squared_deviations_mean[id_column])
        return self.describe.loc['std']

    def init_describe(self, percentiles=None):
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

        fill_zero = np.zeros((len(rows_index), self.number_of_columns))
        describe = pd.DataFrame(data= fill_zero, columns=self.columns, index=rows_index)

        describe.loc['min'] = self.data.iloc[0]
        describe.loc['max'] = self.data.iloc[0]

        return describe


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

        self.ft_count()
        self.ft_mean()
        self.ft_std()
        self.ft_min()
        self.ft_percentiles(percentiles)
        self.ft_max()
        return self.describe