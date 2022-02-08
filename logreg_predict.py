from operator import index
import	numpy	as np
import	pandas	as pd
import	sys

from logistic_regression_class	import LogisticRegression

def print_usage():
	exit(0)

def write_to_file(dictionary, filename):
	try:
		with open(filename, 'wt') as file:
			file.write(str(dictionary))
	except:
		print("Unable to write to file : {}.".format(filename))

def main():
	if len(sys.argv) != 4 :
		print_usage()

	model = LogisticRegression(bias=False, verbose=True)

	data = pd.read_csv(sys.argv[1], index_col=0)
	weights = pd.read_csv(sys.argv[2], index_col=0)
	target = pd.read_csv(sys.argv[3], index_col=0)

	data.dropna(axis=1, how='all', inplace=True)
	numerical_data = data.select_dtypes(include=[np.number])

	data_x = numerical_data.fillna(numerical_data.mean())

	features = (data_x - data_x.mean()) / data_x.std()



	result = model.predict(features, weights)
	result.to_csv('./houses.csv')


	accuracy = model.calculate_accuracy(target, result)
	print('target = ', target)
	print('result = ', result)
	print('Accuracy = ', accuracy)




if __name__ == '__main__' :
	main()
