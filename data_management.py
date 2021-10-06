import pandas as pd
import numpy as np
from describe_methods import *

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

def describe_courses(courses):
	describe_courses = pd.DataFrame(columns= courses.columns)
	describe_courses_count_min_max(courses, describe_courses)
	print(describe_courses)



def main():
	courses = get_courses_data()
	if courses is None:
		exit()
	else:
		describe_courses(courses)


if __name__ == "__main__" :
	main()