from time import time
import pandas as pd

pd.options.display.float_format = "{:.6f}".format

class Tools():

    def timer_func(func):
        def wrap_func(self, *args, **kwargs):
            print(f'Starting {func.__name__!r} function.')
            t1 = time()
            result = func(self, *args, **kwargs)
            t2 = time()
            print(f'Function {func.__name__!r} ended with executiton time = {(t2-t1):.4f}s')
            return result
        return wrap_func

    def	ft_format_percentiles(n_th):
        format_number  = lambda n: "{:.0f}".format(n) if n % 1 else int(n)
        indexes = [id + '%' for id in list(map(str, list(format_number(n * 100) for n in n_th)))]
        invalid_percentile = [n for n in n_th if n > 1 or n < 0]
        if len(invalid_percentile) == 0:
            return (indexes)
        return None

    def structure_percentiles(percentiles):
        # The .5 must exist in the describe method results
        percentiles.append(.5) if .5 not in percentiles else percentiles
        # The percentiles results should be displayed in an ascending order
        percentiles.sort()
        # Format the indexes related to each percentile
        indexes_percentiles = Tools.ft_format_percentiles(percentiles)
        # If invalid data in percentiles (n < 0 or n > 1) return None
        if indexes_percentiles == None:
            print ('Error in given indexes : {}.\nIndexes must be >= 0 and <= 1'.format(percentiles))
            return None
        return indexes_percentiles
