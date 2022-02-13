import	pandas	as pd
import	sys

from logistic_regression_class	import LogisticRegression
from logreg_error_management    import *


def main():
    files = manage_arguments(sys.argv, 'train')
    model = LogisticRegression(bias=False, verbose=True)
    try:
        data = pd.read_csv(sys.argv[1], index_col=0)
        convert_str_data(data, 'Best Hand')
    except:
        print_usage('files', files[0])

    try:
        # Target (Y)
        target = data['Hogwarts House']
        # Classes (How many unique target)
        classes = target.unique()
    except:
        print_usage('data', files[0])
    try:
        # Train and get results
        weights = model.onevsall(data, target, classes)
        weights.to_csv('./weights.csv')
    except:
        print_usage('training')

if __name__ == '__main__' :
    main()