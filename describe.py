import	sys
from	tools import *

from data_class import Data

def print_usage():
    exit(0)

def main():
    if len(sys.argv) != 2 :
        print_usage()

    # Describe method
    data = Data.from_csv(sys.argv[1])
    data_describe = data.ft_describe()
    print(data_describe)

if __name__ == '__main__' :
    main()
