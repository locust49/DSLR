import numpy as np
import pandas as pd

class LogisticRegression:

	def __init__(self, learning_rate=None, epoch=None, bias=None) -> None:
		self.learning_rate = 0.01 if learning_rate == None else learning_rate
		self.epoch = 10000 if epoch == None else epoch
		self.bias = True if bias == None else False

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
		sig_val.tofile('./debug/sig_val', '\n', '%.2f')
		return (sig_val)

	@staticmethod
	def sigmoid(x_value) -> np.floating:
		return (1 / (1 + np.exp(-x_value)))

	def train(  self,
				X_data: pd.DataFrame,
				Y_data: pd.DataFrame,
				verbose=False) -> None:

		if verbose is True : print('Training the data.')

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

		# print('Weights: ', self.weights)



	def onevsall(self, X_values_matrix, Y_values_matrix, classes):
		print('Starting training ...')
		class_weight = { _class : None for _class in classes }
		for index, class_name in enumerate(classes):
			Y_select = Y_values_matrix.copy()
			Y_select[Y_values_matrix == classes[index]] = 1
			Y_select[Y_values_matrix != classes[index]] = 0
			self.train(X_values_matrix, Y_select)
			class_weight[class_name] = self.weights
		print('Training done !')
		return (class_weight)


	def predict(self, X_values_matrix, class_weight):
		if self.bias == True:
			x_bias  = np.ones((X_values_matrix.shape[0], 1))
			X_values_matrix = np.concatenate((X_values_matrix, x_bias), axis=1)

		Y_fit = pd.DataFrame(index=range(0, X_values_matrix.shape[0]), columns=class_weight.keys())

		for class_name in class_weight:
			Y_fit[class_name] = self._hypothesis(X_values_matrix, class_weight[class_name])
		Y_fit['pred'] = Y_fit.idxmax(axis=1)
		# Y_fit['pred'].to_csv('./debug/my_pred.csv')
		return(Y_fit['pred'])