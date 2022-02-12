import	pandas	as pd
import	sys

from logistic_regression_class	import LogisticRegression
from logreg_error_management	import *

def main():
	files = manage_arguments(sys.argv)

	model = LogisticRegression(verbose=True, bias=False)

	try:
		data = pd.read_csv(files[0], index_col=0)
		data.dropna(axis=1, how='all', inplace=True)
		convert_str_data(data, 'Best Hand')

	except:
		print_usage('files', files[0])

	try:
		weights = pd.read_csv(files[1], index_col=0)
	except:
		print_usage('files', files[1])

	try :
		result = model.predict(data, weights)
		result.to_csv('./houses.csv')
	except :
		print_usage('prediction')

	if files[3] == True:
		try:
			target = pd.read_csv(files[2], index_col=0)
			accuracy = model.calculate_accuracy(target, result)
			print('Accuracy = ', accuracy)
		except:
			print_usage('files', files[2])


if __name__ == '__main__' :
	main()
