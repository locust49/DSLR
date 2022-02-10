import	numpy	as np
import	pandas	as pd
import	sys

from logistic_regression_class	import LogisticRegression

def print_usage():
	exit(0)

def main():
	# if len(sys.argv) != 2 :
	# 	print_usage()
	# try:
	model = LogisticRegression(fillna='drop')

	data = pd.read_csv(sys.argv[1], index_col=0)

	# Target (Y)
	target = data['Hogwarts House']

	# Classes (How many unique target)
	classes = target.unique()

	# Train and get results
	weights = model.onevsall(data, target, classes)
	weights.to_csv('./weights.csv')
	# except :
	# 	print('An error somewhere occured ! Reboot maybe ?')
	# 	pass

if __name__ == '__main__' :
	main()