from ast import While
from multiprocessing.sharedctypes import Value
from tkinter.messagebox import NO
import numpy as np

class LogisticRegression():

	def __init__(self, learning_rate=None, epoch=None) -> None:
		self.learning_rate = 0.01 if learning_rate == None else learning_rate
		self.epoch = 10000 if epoch == None else epoch

	# y_predicted_value = self.hypothesis_matrix(X_matrix, bias, weights)
	def loss(self, y_predicted_value, y_value) -> np.floating:
		Y_class_0 = - np.log(y_predicted_value)
		Y_class_1 = - np.log(1 - y_predicted_value)

		return ((y_value * Y_class_0) + ((1 - y_value) * Y_class_1))

	def cost(self, sum_loss, number_of_x) -> np.floating:
		return sum_loss / number_of_x

	def cost_weights_derivative(Y_predicted_matrix, Y_matrix, X_matrix) -> np.matrix:
		number_of_data = Y_matrix.shape[0]

		#### Raise exception for len = 0
		# if number_of_data < 1 :
		# 	raise (ValueError('Incorrect number of data', number_of_data))
		#####################################################################

		return ((1 / number_of_data) * np.dot((Y_predicted_matrix - Y_matrix), X_matrix))

	def cost_bias_derivative() -> np.floating:
		pass

	# def hypothesis(self, x_value, bias, weight) -> None:
	# 	return (self.sigmoid(bias + x_value * weight))

	def hypothesis_matrix(self, X_matrix, bias, weights) -> None:
		return (self.sigmoid(bias + np.dot(X_matrix, weights)))

	@staticmethod
	def sigmoid(x_value) -> np.floating:
		return 1 / 1 + np.exp(0 - x_value)


	def gradient_descent(self, X_matrix, Y_matrix):
		pass

	def train(self, X_matrix, Y_matrix) -> None:

		self.gradient_descent()