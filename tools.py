class Tools():

	def	ft_format_percentiles(n_th):
		format_number  = lambda n: "{:.0f}".format(n) if n % 1 else int(n)
		indexes = [id + '%' for id in list(map(str, list(format_number(n * 100) for n in n_th)))]
		invalid_percentile = [n for n in n_th if n > 1 or n < 0]
		if len(invalid_percentile) == 0:
			return (indexes)
		return None

	def structure_percentiles(percentiles):
		if percentiles == None:
			percentiles = [.25, .5, .75]
		# The .5 must exist in the describe method results
		percentiles.append(.5) if .5 not in percentiles else percentiles
		# The percentiles results should be displayed in an ascending order
		percentiles.sort()
		# Format the indexes related to each percentile
		indexes_percentiles = Tools.ft_format_percentiles(percentiles)
		# If invalid data in percentiles (n < 0 or n > 1) return None
		if indexes_percentiles == None:
			return None
		return indexes_percentiles

	#def fabs():
	#	pass

	#def sqrt():
	#	pass

	#def ceil():
	#	pass