import	numpy as np
import	pandas as pd
from	describe_methods import *

'''
	Get the data from the dataset test
	Get the numerical data only
	Drop the columns filled with NaN
	Return the courses.
'''

def	get_courses_data():
	try:
		data = pd.read_csv('./datasets/dataset_test.csv', index_col=0)
	except:
		print('Unable to find data.')
		return None
	data_numeric = data.select_dtypes(include=[np.number])
	courses = data_numeric.dropna(axis=1, how='all')
	return courses

'''
	Get statistics data from courses
	Implement the count, mean, std,
		percentiles, min and max methodes
'''

def initialize_describe_df(number_of_columns):
	# Create a dataframe filled with 0
	rows_index = ['count',
				'mean',
				'std',
				'min',
				'25%',
				'50%',
				'75%',
				'max',
	]
	zeros = np.zeros((len(rows_index), len(number_of_columns)))
	describe = pd.DataFrame(data=zeros,columns= number_of_columns, index= rows_index)
	return describe

def describe_courses(courses):
	describe_courses = initialize_describe_df(courses.columns)
	describe_courses_count_min_max(courses, describe_courses)
	describe_course_percentiles([25, 50, 75], courses, describe_courses)
	describe_course_std(describe_courses, courses)
	# print('EXPECTED')
	# print(courses.describe())
	# print('\nRESULT')
	print(describe_courses)


def main():
	courses = get_courses_data()
	if courses is None:
		exit()
	else:
		describe_courses(courses)


if __name__ == "__main__" :
	main()