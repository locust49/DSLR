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
	if len(sys.argv) != 2 :
		print_usage()
	try:
		model = LogisticRegression(bias=False, verbose=True)

		data = pd.read_csv(sys.argv[1], index_col=0)

		# Features (X)
		numerical_data = data.select_dtypes(include=[np.number])
		features = numerical_data.fillna(numerical_data.mean())
		features_normalized = (features - features.mean()) / features.std()

		# Target (Y)
		target = data['Hogwarts House']

		# Classes (How many unique target)
		classes = target.unique()

		# Train and get results
		weights = model.onevsall(features_normalized, target, classes)
		weights.to_csv('./weights.csv')
	except :
		print('An error somewhere occured ! Reboot maybe ?')
		pass

if __name__ == '__main__' :
	main()