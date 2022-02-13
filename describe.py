import	sys
from	tools import *

from    data_class import Data
from    logreg_error_management import print_usage

def main():
    if len(sys.argv) != 2 :
       print_usage('describe', sys.argv[0])

    try:
        data = Data.from_csv(sys.argv[1])
    except:
        print_usage('files', sys.argv[1])

    # Describe method
    data_describe = data.ft_describe()
    print(data_describe)

if __name__ == '__main__' :
    main()
