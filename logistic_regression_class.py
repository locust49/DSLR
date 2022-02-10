from lzma import PRESET_DEFAULT
import numpy as np
import pandas as pd
from time import time

class LogisticRegression:

    def __init__(self, learning_rate=None, epoch=None, bias=None, fillna=None, normalize=None, verbose=None) -> None:
        self.learning_rate = 0.01 if learning_rate == None else learning_rate
        self.epoch = 10000 if epoch == None else epoch
        self.bias = True if bias == None else bias
        self.fillna = 'mean' if fillna == None else fillna
        self.verbose = False if verbose == None else verbose
        self.normalize = True if normalize == None else normalize

    def verbose_func(func):
        def wrap_func(self, *args, **kwargs):
            if self.verbose is True: print(f'Starting {func.__name__!r} function.')
            t1 = time()
            result = func(self, *args, **kwargs)
            t2 = time()
            if self.verbose is True: print(f'Function {func.__name__!r} ended with executiton time = {(t2-t1):.4f}s')
            return result
        return wrap_func

    def loss(self, y_predicted_value, y_value) -> np.float64:
        Y_class_1 =  np.log(y_predicted_value)
        Y_class_0 =  np.log(1 - y_predicted_value)
        loss_value = np.dot(y_value, Y_class_0) + (np.dot((1 - y_value), Y_class_1))
        return np.float64(loss_value)

    def cost_derivative_weights(self, Y_predicted_matrix, Y_matrix, X_matrix) -> np.matrix:
        number_of_data = X_matrix.shape[0]
        return ((1 / number_of_data) * np.dot((Y_predicted_matrix - Y_matrix).transpose() , X_matrix))

    def cost_derivative_bias(self, Y_predicted_matrix, Y_matrix) -> np.floating:
        number_of_data = Y_matrix.shape[1]
        return ((1 / number_of_data) * np.sum(Y_predicted_matrix - Y_matrix))

    def _hypothesis(self, X_matrix, weights=None) -> np.floating:

        if type(weights) == type(None) :
            weights = self.weights
        X_value = np.dot(X_matrix, weights.transpose())
        X_value_float = np.array(X_value, dtype=np.floating)
        sig_val = self.sigmoid(X_value_float)
        return (sig_val)

    @staticmethod
    def sigmoid(x_value) -> np.floating:
        return (1 / (1 + np.exp(-x_value)))

    @staticmethod
    def fill_nan_data(data, fill_value):
        numerical_data = data.select_dtypes(include=[np.number])

        if fill_value == 'max':
            new_data = numerical_data.fillna(numerical_data.max())
        elif fill_value == 'mean':
            new_data = numerical_data.fillna(numerical_data.mean())
        elif fill_value == 'median':
            new_data = numerical_data.fillna(numerical_data.median())
        elif fill_value == 'min':
            new_data = numerical_data.fillna(numerical_data.min())
        elif fill_value == 'std':
            new_data = numerical_data.fillna(numerical_data.std())
        elif fill_value == 'zero':
            new_data = numerical_data.fillna(0.)
        elif fill_value == 'drop':
            new_data = numerical_data.dropna()
        else:
            new_data = None
            raise Exception('Undefined fill value : {}'.format(fill_value))
        return new_data

    @staticmethod
    def normalize_data(data):
        numerical_data = data.select_dtypes(include=[np.number])
        normalized_data = (numerical_data - numerical_data.mean()) / numerical_data.std()
        return (normalized_data)

    def clean(  self,
                data : pd.DataFrame
                ) -> pd.DataFrame:

        new_data = self.fill_nan_data(data, self.fillna)
        if self.normalize == True:
            return self.normalize_data(new_data)
        return new_data

    def train(  self,
                X_data: pd.DataFrame,
                Y_data: pd.DataFrame,
                ) -> None:

        X_data = self.clean(X_data)
        X_values_matrix = np.matrix(X_data.to_numpy())
        Y_values_matrix = np.matrix(Y_data.to_numpy())
        if self.bias == True:
            X_bias  = np.ones((X_values_matrix.shape[0], 1))
            X_values_matrix = np.concatenate((X_values_matrix, X_bias), axis=1)

        total_data, total_weights = X_values_matrix.shape
        self.weights = np.matrix([1.0 for i in range(total_weights)])

        Y_predicted_matrix = self._hypothesis(X_values_matrix)
        for _ in range(self.epoch):
            gradient_weights = self.cost_derivative_weights(Y_predicted_matrix, Y_values_matrix.transpose(), X_values_matrix)
            self.weights = self.weights - np.multiply(self.learning_rate, gradient_weights)
            Y_predicted_matrix = self._hypothesis(X_values_matrix)

        return self.weights


    @verbose_func
    def onevsall(self, X_data, Y_data, classes):
        X_values_matrix = self.clean(X_data)
        class_weight_df_columns = X_values_matrix.columns.tolist()
        if self.bias == True:
            class_weight_df_columns.append('Bias')

        class_weight_df = pd.DataFrame(index=classes, columns=class_weight_df_columns)
        class_weight_df.index.name = Y_data.name
        for index, class_name in enumerate(classes):
            Y_select = Y_data.copy()
            Y_select[Y_data == classes[index]] = 1
            Y_select[Y_data != classes[index]] = 0
            if self.fillna == 'drop' :
                indexes = X_values_matrix.index.intersection(Y_select.index)
                Y_select = Y_select.iloc[indexes]
            weights = self.train(X_values_matrix, Y_select)
            class_weight_df.loc[class_name] = weights
        return class_weight_df

    @verbose_func
    def predict(self, X_data, weights) -> pd.DataFrame:
        X_data = self.clean(X_data)
        if self.bias == True:
            X_bias  = np.ones((X_data.shape[0], 1))
            X_values_matrix = np.concatenate((X_data, X_bias), axis=1)

        Y_fit = pd.DataFrame(index=range(0, X_values_matrix.shape[0]), columns=weights.index)
        Y_fit.index.name = 'Index'

        for class_name in weights.index:
            Y_fit[class_name] = self._hypothesis(X_values_matrix, weights.loc[class_name])
        Y_fit[weights.index.name] = Y_fit.idxmax(axis=1)
        return(Y_fit[[weights.index.name]])

    @staticmethod
    def calculate_accuracy(target_Y, predicted_Y):
        accuracy = np.where(predicted_Y == target_Y , 1, 0)
        return (str(accuracy.mean() * 100) + '%')
