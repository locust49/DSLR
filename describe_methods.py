import numpy as np
import pandas as pd
import math

'''
	count	method:
		counts the number of non-NA/null observations.
	min		method:
		returns the minimal value of the observations.
	max		method:
		returns the maximal value of the observations.
'''

def describe_courses_count_min_max(courses, describe_courses):
	number_of_columns = len(describe_courses.columns)

	describe_courses.loc['min'] = courses.iloc[0]
	describe_courses.loc['max'] = courses.iloc[0]

	list_sum = [0] * number_of_columns

	for observation in courses.iterrows():
		for i in range(number_of_columns):

			# Calculate count and sum
			if observation[1][i] is not None and not np.isnan(observation[1][i]):
				describe_courses.loc['count'][i] += 1
				list_sum[i] += observation[1][i]
			else:
				list_sum[i] += 0

			# Calculate min
			if observation[1][i] <= describe_courses.loc['min'][i]:
				describe_courses.loc['min'][i] = observation[1][i]

			# Calculate max
			if observation[1][i] >= describe_courses.loc['max'][i]:
				describe_courses.loc['max'][i] = observation[1][i]

	for i in range(number_of_columns):
		if describe_courses.loc['count'][i] == 0 :
			describe_courses.loc['mean'][i] = np.nan
		else:
			describe_courses.loc['mean'][i] = list_sum[i] / describe_courses.loc['count'][i]

	return (describe_courses)



'''
	std		method:
		The `standard deviation` is the square root of the average of the squared deviations from the mean
		i.e., `std = sqrt(mean(x))`, where `x = abs(a - a.mean())**2`.
'''

### Insert some code here.

'''
	percentiles:
		n-th %:

'''

def describe_course_percentiles(n_th, courses, describe_courses):
	number_of_columns = len(describe_courses.columns)
	minus, plus = {}, {}

	for i in range(number_of_columns):
		sorted_courses = courses.sort_values(by=[courses.columns[i]])
		index = n_th * describe_courses.loc['count'][i] / 100
		if index % 2 == 0:
			describe_courses.loc['25%'][i] = sorted_courses.iloc[int(index)][i]
			minus[courses.columns[i]] = sorted_courses.iloc[int(index)][i]
			plus[courses.columns[i]] = sorted_courses.iloc[int(index)][i]
		else:
			# index = int(math.ceil(index))
			# if (index > 1):
			# describe_courses.loc['25%'][i] = (sorted_courses.iloc[int(math.ceil(index)) - 1][i] + sorted_courses.iloc[int(math.ceil(index))][i])/2
			describe_courses.loc['25%'][i] = sorted_courses.iloc[int(math.ceil(index)) - 1][i]
			# elif (index < describe_courses.loc['count'][i]):
				# describe_courses.loc['25%'][i] = (sorted_courses.iloc[index + 1][i] + sorted_courses.iloc[index][i])/2

			# minus[courses.columns[i]] = (sorted_courses.iloc[index - 1][i] + sorted_courses.iloc[index][i])/2
			# plus[courses.columns[i]] = (sorted_courses.iloc[index + 1][i] + sorted_courses.iloc[index][i])/2

	# print(pd.Series(minus), pd.Series(plus), '\n',courses.describe().loc['25%'])
	print(describe_courses)