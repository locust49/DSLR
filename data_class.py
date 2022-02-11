from calendar import c
import	numpy	as np
from	numpy.lib	import percentile
from	tools		import *
import	pandas	as pd
import	math

class Data:
    def __init__(self, dataframe: pd.DataFrame, normalize=False):
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

        self.columns = list(self.numerical_data.columns) if not self.numerical_data.empty else list(self.initial_data.columns)

        if not self.numerical_data.empty :
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

        data = self.initial_data if self.numerical_data.empty else self.numerical_data
        count = { key:float(0) for key in self.columns }

        for index, row in data.iterrows():
            for column in self.columns:
                if type(row[column]) == str:
                    if row[column]:
                        count[column] += 1
                else :
                    if row[column] is not None :
                        if not np.isnan(row[column]):
                                count[column] += 1

        return count

    def ft_unique(self):
        '''
        unique	method:
            counts the number of unique rows values.
        '''

        unique = { key:float(0) for key in self.columns }
        values = {column_name: [] for column_name in self.columns}

        for index, row in self.initial_data.iterrows():
            for column in self.columns:
                if row[column] not in values[column]:
                    values[column].append(row[column])
                unique[column] = len(values[column])
        return unique

    def ft_freq(self):
        '''
        freq method:
            returns the most repeated value.
        '''

        freq = { key:float(0) for key in self.columns }
        values = {column_name: {} for column_name in self.columns}

        for index, row in self.initial_data.iterrows():
            for column in self.columns:
                if row[column] not in values[column]:
                    values[column].update({row[column]: 1})
                else:
                    values[column][row[column]] += 1
        for column in self.columns:
            freq[column] = sorted(values[column].items(), key=lambda item:item[1])[-1][1]
        return freq

    def ft_top(self):
        '''
        top	method:
            returns the most repeated value.
        '''

        top = { key:float(0) for key in self.columns }
        values = {column_name: {} for column_name in self.columns}

        for index, row in self.initial_data.iterrows():
            for column in self.columns:
                if row[column] not in values[column]:
                    values[column].update({row[column]: 0})
                else:
                    values[column][row[column]] += 1
        for column in self.columns:
            top[column] = sorted(values[column].items(), key=lambda item:item[1])[-1][0]
        return top


    def ft_min(self):
        '''
        min		method:
            returns the minimal value of the rows.
        '''
        if self.numerical_data.empty:
            return None
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
        if self.numerical_data.empty:
            return None
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
        if self.numerical_data.empty:
            return None
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
        if self.numerical_data.empty:
            return None
        mean = { key:float(0) for key in self.columns }

        for column in self.columns:
            if self.counted_data[column] == 0 :
                mean[column] = np.nan
            else:
                mean[column] = float(self.summed_data[column]) / float(self.counted_data[column])
        return mean

    def	ft_percentiles(self, n_th=None):
        '''
        percentiles	method:
            Returns the n-th percentile(s) of the rows.
            n_th:
                The default is [.25, .5, .75], which returns the 25th, 50th, and 75th percentiles.
                All should fall between 0 and 1.
        '''
        if self.numerical_data.empty:
            return None
        if n_th is None :
            n_th = [.25, .5, .75]

        # Format the indexes (input : .25 | output : 25%)
        indexes = Tools.ft_format_percentiles(n_th)

        if indexes == None:
            print ('Error in given indexes : {}.\nIndexes must be >= 0 and <= 1'.format(n_th))
            exit(0)

        percentiles_data = { index:{ key:float(0) for key in self.columns } for index in indexes}

        for n_percentile, id_percentile in zip(n_th, indexes):
            for id_column, column in enumerate(self.columns):
                # Sort the data by column
                sorted_data = self.numerical_data.sort_values(by=[self.columns[id_column]])
                # Compute the index of the percentile
                index = float(n_percentile * (int(self.counted_data[column]) - 1))

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
        if self.numerical_data.empty:
            return None
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

    @Tools.timer_func
    def ft_describe(self, percentiles=None):
        '''
            Implementation of the DataFrame.describe method.
            Returns a dataframe of the 'count', 'mean', 'std',
                'min', 'percentiles', and 'max' methods
        '''
        if percentiles is None:
            percentiles = [.25, .5, .75]

        indexes_percentiles = Tools.structure_percentiles(percentiles)
        if indexes_percentiles == None:
            return None

        # Initialize the result dataframe.
        if self.numerical_data.empty:
            rows_index = ['count',
                    'unique',
                    'top',
                    'freq',
            ]
        else:
            rows_index = ['count',
                        'mean',
                        'std',
                        'min',
                        *indexes_percentiles,
                        'max',
            ]

        describe = pd.DataFrame(columns=self.columns, index=rows_index)

        describe.loc['count'] = self.ft_count()
        if self.numerical_data.empty:
            describe.loc['unique'] = self.ft_unique()
            describe.loc['top'] = self.ft_top()
            describe.loc['freq'] = self.ft_freq()
        if not self.numerical_data.empty:
            describe.loc['mean'] = self.ft_mean()
            describe.loc['std'] = self.ft_std()
            describe.loc['min'] = self.ft_min()
            percentiles_results = self.ft_percentiles(percentiles)
            for n_th in indexes_percentiles:
                describe.loc[n_th] = percentiles_results[n_th]
            describe.loc['max'] = self.ft_max()

        return describe