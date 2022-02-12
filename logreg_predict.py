import	pandas	as pd
import	sys
import	numpy	as np


from logistic_regression_class	import LogisticRegression

def print_usage(error_string, *args):
	if error_string == 'label':
		print('Invalid labels.\nMissing ', *args)
	if error_string == 'extension':
		print('Invalid extension for :', *args)
	if error_string == 'target-file':
		print('Target file missing while specifiying accuracy')
	if error_string == 'idk':
		print('How did you get here ?')
	if error_string == 'prediction':
		print('Error in prediction !\nInvalid data or source file data !')
	if error_string == 'files':
		print('Error while opening the file : ', *args)
	exit(0)


def check_if_all_labelled(args):
	label_counter = 0
	for arg in args:
		if arg[0:2] == '--':
			label_counter += 1
		if arg == '-with-accuracy':
			continue
	if label_counter == 0:
		return False
	return True


def check_if_good(files):
	keys = []
	values = []
	for key, val in files.items():
		if val == '':
			if key == 'dataset-target' and files['accuracy'] == False:
				continue
			keys.append(key)
		elif type(val) == type('') and len(val) > 4 and val[-4:] != '.csv':
			values.append(val)

	if len(keys) != 0:
		print_usage('label', keys)
		return False
	if  type(val) == type('') and len(values) != 0:
		print_usage('extension', values)
		return False
	return True


def manage_arguments(args):
	files = {	'dataset-test' : '',
				'dataset-weights' : '',
				'dataset-target' : '',
				'accuracy' : False
			}
	if '-with-accuracy' in args:
		files['accuracy'] = True
		args.remove('-with-accuracy')
	if check_if_all_labelled(args) :
		for arg in args:
			# if arg == '-with-accuracy':
			# 	continue
			if arg[0:len('--accuracy-target=')] == '--accuracy-target=':
				files['dataset-target'] = arg[len('--accuracy-target='):]
			if arg[0:len('--dataset-weights=')] == '--dataset-weights=':
				files['dataset-weights'] = arg[len('--dataset-weights='):]
			if arg[0:len('--dataset-test=')] == '--dataset-test=':
				files['dataset-test'] = arg[len('--dataset-test='):]
	else:
		files['dataset-test'] = args[1]
		files['dataset-weights'] = args[2]
		try:
			if files['accuracy'] == True:
				files['dataset-target'] = args[3]
		except:
			print_usage('target-file')


	if check_if_good(files):
		return list(files.values())

	print_usage('idk')


def convert_str_data(data, column_name):
	data_column = data[column_name]
	values = data_column.unique()
	data_column_num = data_column.copy()
	for index, val in enumerate(values):
		data_column_num[data_column == val] = index
	data[column_name] = data_column_num.astype(np.integer)


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
