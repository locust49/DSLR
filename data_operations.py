import pandas as pd

class data_operations:
	def __init__(self, data_source='./datasets/dataset_train.csv'):
		self.dataframe = pd.read_csv(data_source)
		self.init_constants()


	def init_constants(self):
		self.labels = self.dataframe.columns.values
		self.classes = self.labels[6:]
		self.houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
		self.total = len(self.dataframe)

	def get_data(self):
		return (self.dataframe)


if __name__ == '__main__':
	data_ops = data_operations()
	print (data_ops.classes)