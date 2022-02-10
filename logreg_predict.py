from operator import index
import	numpy	as np
import	pandas	as pd
import	sys

from logistic_regression_class	import LogisticRegression

def print_usage():
	exit(0)

def main():
	if len(sys.argv) != 4 :
		print_usage()

	model = LogisticRegression()

	data = pd.read_csv(sys.argv[1], index_col=0)
	weights = pd.read_csv(sys.argv[2], index_col=0)
	target = pd.read_csv(sys.argv[3], index_col=0)

	data.dropna(axis=1, how='all', inplace=True)

	result = model.predict(data, weights)
	result.to_csv('./houses.csv')

	accuracy = model.calculate_accuracy(target, result)
	print('Accuracy = ', accuracy)

if __name__ == '__main__' :
	main()
