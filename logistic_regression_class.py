from lzma import PRESET_DEFAULT
import numpy as np
import pandas as pd
from time import time

class LogisticRegression:

    def __init__(self, learning_rate=None, epoch=None, bias=None, verbose=None) -> None:
        self.learning_rate = 0.01 if learning_rate == None else learning_rate
        self.epoch = 10000 if epoch == None else epoch
        self.bias = True if bias == None else False
        self.verbose = False if verbose == None else True

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

    def train(  self,
                X_data: pd.DataFrame,
                Y_data: pd.DataFrame,
                ) -> None:

        X_values_matrix = np.matrix(X_data.to_numpy())
        Y_values_matrix = np.matrix(Y_data.to_numpy())
        if self.bias == True:
            x_bias  = np.ones((X_values_matrix.shape[0], 1))
            X_values_matrix = np.concatenate((X_values_matrix, x_bias), axis=1)

        total_data, total_weights = X_values_matrix.shape
        self.weights = np.matrix([1.0 for i in range(total_weights)])

        Y_predicted_matrix = self._hypothesis(X_values_matrix)
        for _ in range(self.epoch):
            gradient_weights = self.cost_derivative_weights(Y_predicted_matrix, Y_values_matrix.transpose(), X_values_matrix)
            self.weights = self.weights - np.multiply(self.learning_rate, gradient_weights)
            Y_predicted_matrix = self._hypothesis(X_values_matrix)

        return self.weights


    @verbose_func
    def onevsall(self, X_values_matrix, Y_values_matrix, classes):
        class_weight_df_columns = X_values_matrix.columns.tolist()
        if self.bias == True:
            class_weight_df_columns.append('Bias')

        class_weight_df = pd.DataFrame(index=classes, columns=class_weight_df_columns)
        class_weight_df.index.name = Y_values_matrix.name
        for index, class_name in enumerate(classes):
            Y_select = Y_values_matrix.copy()
            Y_select[Y_values_matrix == classes[index]] = 1
            Y_select[Y_values_matrix != classes[index]] = 0
            weights = self.train(X_values_matrix, Y_select)
            class_weight_df.loc[class_name] = weights
        return class_weight_df

    @verbose_func
    def predict(self, X_values_matrix, weights) -> pd.DataFrame:
        if self.bias == True:
            x_bias  = np.ones((X_values_matrix.shape[0], 1))
            X_values_matrix = np.concatenate((X_values_matrix, x_bias), axis=1)

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
