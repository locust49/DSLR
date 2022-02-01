import numpy as np
import pandas as pd

from maths_class import Mathematics

def write(this):
	with open('./debug/file.txt', 'w+') as f :
		f.write(this.toString())

class LogisticRegression:

	def __init__(self, learning_rate=None, epoch=None, bias=None) -> None:
		self.learning_rate = 0.01 if learning_rate == None else learning_rate
		self.epoch = 10000 if epoch == None else epoch
		self.bias = True if bias == None else False

	def loss(self, y_predicted_value, y_value) -> np.float64:

		y_predicted_value.tofile('./debug/y_pred', '\n', '%.2f')
		y_value.tofile('./debug/y_val', '\n', '%.2f')

		# print('y_predicted_value: ', type(y_predicted_value), y_predicted_value.shape)
		# print('1 - y_predicted_value: ', 1 - y_predicted_value)

		Y_class_1 =  np.log(y_predicted_value)
		Y_class_0 =  np.log(1 - y_predicted_value)
		Y_class_0.tofile('./debug/Y_class_0', '\n', '%.2f')
		Y_class_1.tofile('./debug/Y_class_1', '\n', '%.2f')
		# print('Y_class_0: ', type(Y_class_0), Y_class_0.shape, Y_class_0)
		# print('Y_class_1: ', type(Y_class_1), Y_class_1.shape, Y_class_1)

		# print('y_value: ', type(y_value), y_value.shape)


		loss_value = (np.dot(y_value, Y_class_0) + (np.dot((1 - y_value), Y_class_1)))
		# print('y_value: ', type(y_value), y_value.shape)

		# print('loss_value ', np.float64(loss_value))
		return np.float64(loss_value)


	def cost_derivative_weights(self, Y_predicted_matrix, Y_matrix, X_matrix) -> np.matrix:
		number_of_data = X_matrix.shape[0]

		#print('Y_predicted_matrix: ', type(Y_predicted_matrix), Y_predicted_matrix.shape)
		#print('Y_matrix: ', type(Y_matrix), Y_matrix.shape)
		#print('(Y_predicted_matrix - Y_matrix).transpose(): ', type((Y_predicted_matrix.transpose() - Y_matrix).transpose()), (Y_predicted_matrix - Y_matrix).transpose().shape)

		return ((1 / number_of_data) * np.dot((Y_predicted_matrix - Y_matrix).transpose() , X_matrix))

	def cost_derivative_bias(self, Y_predicted_matrix, Y_matrix) -> np.floating:
		number_of_data = Y_matrix.shape[1]
		return ((1 / number_of_data) * (Y_predicted_matrix - Y_matrix))

	def _hypothesis(self, X_matrix) -> np.floating:
		X_value = np.dot(X_matrix, self.weights.transpose())
		X_value_float = np.array(X_value, dtype=np.floating)
		sig_val = self.sigmoid(X_value_float)
		# print(sig_val)
		# X_value_float.tofile('./debug/X_value_float', '\n', '%.2f')
		sig_val.tofile('./debug/sig_val', '\n', '%.2f')
		return (sig_val)

	def hypothesis(self, X_matrix, weights) -> np.floating:
		print('w = {}, {}'.format(type(weights), weights))
		X_value = np.dot(X_matrix, weights.transpose())
		X_value_float = np.array(X_value, dtype=np.floating)
		sig_val = self.sigmoid(X_value_float)
		# print(sig_val)
		# X_value_float.tofile('./debug/X_value_float', '\n', '%.2f')
		sig_val.tofile('./debug/sig_val', '\n', '%.2f')
		return (sig_val)

	@staticmethod
	def sigmoid(x_value) -> np.floating:
		# # print('x_value: ', type(x_value), x_value.shape)
		# s = np.sum(x_value)
		# # print('s: ', type(s))

		return (1 / (1 + np.exp(-x_value)))

	def train(  self,
				X_data: pd.DataFrame,
				Y_data: pd.DataFrame,
				verbose=False) -> None:

		X_values_matrix = np.matrix(X_data.to_numpy())
		Y_values_matrix = np.matrix(Y_data.to_numpy())
		if self.bias == True:
			x_bias  = np.ones((X_values_matrix.shape[0], 1))
			X_values_matrix = np.concatenate((X_values_matrix, x_bias), axis=1)

		total_data, total_weights = X_values_matrix.shape
		self.weights = np.matrix([1.0 for i in range(total_weights)])

		Y_predicted_matrix = self._hypothesis(X_values_matrix)
		for _ in range(self.epoch):
			#print('Y_predicted_matrix: ', type(Y_predicted_matrix), Y_predicted_matrix.shape)
			#print('Y_values_matrix: ', type(Y_values_matrix), Y_values_matrix.shape)
			gradient_weights = self.cost_derivative_weights(Y_predicted_matrix, Y_values_matrix.transpose(), X_values_matrix)
			# print(self.weights.shape, gradient_weights.shape)
			#print('weight: ', type(self.weights), self.weights.shape)
			#print('gradient_weights: ', type(gradient_weights), gradient_weights.shape)
			self.weights = self.weights - np.multiply(self.learning_rate, gradient_weights)

			#print('X_values_matrix: ', type(X_values_matrix), X_values_matrix.shape)
			Y_predicted_matrix = self._hypothesis(X_values_matrix)

		print('Weights: ', type(self.weights), self.weights.shape)



	def onevsall(self, X_values_matrix, Y_values_matrix, classes):
		class_weight = { _class : None for _class in classes }
		for index, class_name in enumerate(classes):
			Y_select = Y_values_matrix.copy()
			Y_select[Y_values_matrix == classes[index]] = 1
			Y_select[Y_values_matrix != classes[index]] = 0
			self.train(X_values_matrix, Y_select)
			# print('loss shape = {}'.format(loss.shape))
			class_weight[class_name] = self.weights.astype(np.float64)
			# break
		# print('costs = {}, max_cost = {} {}'.format(costs, type(np.max(costs)), np.max(costs)))
		# index = costs.index(np.max(costs))
		# print('index = {}'.format(index))
		# self.weights = class_weight[index]
		# self.weights.tofile('./debug/weights', '\n', '%.2f')
		print('class_weight = {}'.format(class_weight))
		return (class_weight)


	def predict(self, X_values_matrix, class_weight):
		if self.bias == True:
			x_bias  = np.ones((X_values_matrix.shape[0], 1))
			X_values_matrix = np.concatenate((X_values_matrix, x_bias), axis=1)

		Y_fit = pd.DataFrame(index=range(0, X_values_matrix.shape[0]), columns=class_weight.keys())

		for class_name in class_weight:
			Y_fit[class_name] = self.hypothesis(X_values_matrix, class_weight[class_name])
		Y_fit['pred'] = Y_fit.idxmax(axis=1)


		Y_fit['pred'].to_csv('./debug/my_pred.csv')
		# print(Y_fit)

		# class_weight.tofile('./debug/result', '\n', '%f')
		return(Y_fit['pred'])