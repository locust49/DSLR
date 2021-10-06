import numpy as np

'''
	count method:
		counts the number of non-NA/null observations.
'''

def describe_courses_count_min_max(courses, describe_courses):
	number_of_columns = len(describe_courses.columns)

	describe_courses.loc['count'] = [0] * number_of_columns
	describe_courses.loc['mean'] = [0] * number_of_columns
	describe_courses.loc['std'] = [0] * number_of_columns
	describe_courses.loc['min'] = courses.iloc[0]
	describe_courses.loc['25%'] = [0] * number_of_columns
	describe_courses.loc['50%'] = [0] * number_of_columns
	describe_courses.loc['75%'] = [0] * number_of_columns
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
	std method:
		The `standard deviation` is the square root of the average of the squared deviations from the mean
		i.e., `std = sqrt(mean(x))`, where `x = abs(a - a.mean())**2`.
'''

