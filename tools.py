class Tools():

	def	ft_format_percentiles(n_th):
		format_number  = lambda n: "{:.2f}".format(n) if n % 1 else int(n)
		indexes = [id + '%' for id in list(map(str, list(format_number(n * 100) for n in n_th)))]
		invalid_percentile = [n for n in n_th if n > 1 or n < 0]
		if len(invalid_percentile) == 0:
			return (indexes)
		return None

	#def fabs():
	#	pass

	#def sqrt():
	#	pass

	#def ceil():
	#	pass