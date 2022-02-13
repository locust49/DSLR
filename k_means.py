import math
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from collections import Counter

mean_id = 0

class mean:
    def __init__(self, dimension=2, id=None):
        global mean_id
        self.dimension = dimension
        self.pos = [0 for _ in range(dimension)]
        self.closest_points = []
        self.closest_types = []
        self.cluster_name = None
        if id == None:
            self.id = mean_id
        else:
            self.id = id
        mean_id += 1

    def set_pos(self, *argv):
        if len(argv) != self.dimension:
            raise Exception("Position set is of wrong dimensions")
        new_pos = []
        try:
            for arg in argv:
                new_pos.append(float(arg))
        except ValueError:
            raise Exception("set_pos parameters should be float")
        self.pos = new_pos
    
    def reposition(self):
        mean_pos = [0 for _ in range(self.dimension)]
        types_count = Counter(self.closest_types)
        self.cluster_name = list(types_count)[0]
        for point in self.closest_points:
            for i in range(self.dimension):
                mean_pos[i] += point[i]
        for i in range(self.dimension):
            mean_pos[i] /= len(self.closest_points)
        if mean_pos == self.pos:
            return False
        self.pos = mean_pos
        return True

    def clear_closest_points(self):
        self.closest_points = []
        return (self)
    
    def clear_closest_types(self):
        self.closest_types = []
        return (self)
    
    def calc_distance(self, point):
        distance_sq = 0
        for i in range(self.dimension):
            distance_sq += (self.pos[i] - point[i])**2
        # to make things faster, we can use squared distance, by removing the sqrt
        return math.sqrt(distance_sq)

    def set_close_point(self, point):
        self.closest_points.append(point)
        return (self)
    
    def set_close_type(self, point_type):
        self.closest_types.append(point_type)
        return (self)

    def __str__(self):
        return ("<mean " + str(self.pos) + ">")
    
    def __repr__(self):
        return self.__str__()

class k_means:
    def __init__(self, k=4, dimension=2):
        self.k = k
        self.dimension = dimension
        self.means = [mean(dimension, id=i) for i in range(k)]
        self.points = []
        self.points_type = []
        self.total_iterations = 0
        self.type_prop = ""

    def normalize_dataset(self, dataset, props):
        for column in props:
            dataset[column] = ((dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())) * 1000
        return dataset

    def train_init(self, data_frame, props, type_prop):
        # get initialize the k means
        self.type_prop = type_prop
        #data_frame = data_frame.dropna(subset=props)
        data_frame = data_frame.fillna(data_frame.mean())
        initial_mean_positions = data_frame.sample(n=self.k)[props]
        mean_index = 0
        for _, row in initial_mean_positions.iterrows():
            mean = self.means[mean_index]
            new_pos = []
            for prop in props:
                new_pos.append(row[prop])
            mean.set_pos(*new_pos)
            mean_index += 1
        # creating points from dataframe
        for _, row in data_frame.iterrows():
            self.points.append(self.entry_to_pos(row, props))
            self.points_type.append(row[self.type_prop])

    def train_iter(self):
        progress = False
        # an iterations training the algorithm
        for mean in self.means:
            mean.clear_closest_points().clear_closest_types()
        for i in range(len(self.points)):
            point = self.points[i]
            point_type = self.points_type[i]
            self.get_closest_mean(point).set_close_point(point).set_close_type(point_type)
        for mean in self.means:
            res = mean.reposition()
            progress |= res
        return (progress)


    def train(self, data_frame, props, type_prop, max_iterations=None):
        if len(props) != self.dimension:
            raise Exception("Number of properties does not match initialized dimension")
        data_frame = self.normalize_dataset(data_frame, props)
        self.train_init(data_frame, props, type_prop)
        iterations = 0
        while (iterations < max_iterations or max_iterations == None) and self.train_iter():
            iterations += 1
        self.total_iterations = iterations
    
    def predict(self, data_frame, props, res_col):
        # maybe you would want to predict even values with NaN
        data_frame.dropna(subset=props, inplace=True)#data_frame[props].fillna(data_frame[props].mean())
        data_frame = self.normalize_dataset(data_frame, props)
        for index, row in data_frame.iterrows():
            pos = self.entry_to_pos(row, props)
            mean = self.get_closest_mean(pos)
            if mean:
                data_frame.loc[index, res_col] = mean.cluster_name
        return (data_frame)


    def get_closest_mean(self, pos):
        closest_dist = math.inf
        closest_mean = None
        for mean in self.means:
            distance = mean.calc_distance(pos)
            if (distance < closest_dist):
                closest_dist = distance
                closest_mean = mean
        return closest_mean

    def entry_to_pos(self, entry, props):
        pos = []
        for prop in props:
            pos.append(entry[prop])
        return pos
    
    def min_inter_cluster_dist(self):
        min_mean = math.inf
        for mean in self.means:
            mean_distance = 0
            for point in mean.closest_points:
                mean_distance += mean.calc_distance(point)
            min_mean = min(min_mean, mean_distance)
        return (min_mean)
    
    def max_intra_cluster_dist(self):
        max_mean = -math.inf
        for mean1 in self.means:
            for mean2 in self.means:
                if mean1 != mean2:
                    max_mean = max(mean1.calc_distance(mean2.pos), max_mean)
        return (max_mean)

    def dunn_index(self):
        min_inter = self.min_inter_cluster_dist()
        max_intra = self.max_intra_cluster_dist()
        return (min_inter/max_intra)

    def plot(self):
        for mean in self.means:
            x = []
            y = []
            for point in self.points:
                if self.get_closest_mean(point).id == mean.id:
                    x.append(point[0])
                    y.append(point[1])
            plt.scatter(x, y)
            plt.scatter([mean.pos[0],], [mean.pos[1],])
        plt.show()

    def __str__(self):
        res = "<k_means " + str(self.means)
        if self.total_iterations != 0:
            res += ", iterations : %s" % (self.total_iterations)
            res += ", score : %f" % (self.dunn_index())
        res += ">"
        return res
    
    def __repr__(self):
        return self.__str__()

class k_means_optimiser():
    def __init__(self, k=4, dimension=2, iterations=10):
        self.k = k
        self.dimension = dimension
        self.iterations = iterations
        self.classifiers = [k_means(k=k, dimension=dimension) for _ in range(iterations)]
        self.best_classifier = None
    
    def train(self, data_frame, props, type_prop, max_iterations=None):
        best_score = -math.inf
        for i in range(self.iterations):
            self.classifiers[i].train(data_frame, props, type_prop, max_iterations=max_iterations)
            score = self.classifiers[i].dunn_index()
            if score > best_score:
                best_score = score
                self.best_classifier = self.classifiers[i]
        return (self.best_classifier)
    
    def predict(self, data_frame, props, res_col):
        if not self.best_classifier:
            return None
        return self.best_classifier.predict(data_frame, props, res_col)
    
    def plot(self):
        if not self.best_classifier:
            return None
        self.best_classifier.plot()
        return self

if __name__ == "__main__":
    from scatter_plot import draw_scatter_plot
    from sklearn.metrics import accuracy_score
    from data_operations import *
    import sys

    if len(sys.argv) == 3:
        input_columns = [sys.argv[1], sys.argv[2]]
    else:
        input_columns = ['Defense Against the Dark Arts', 'Herbology']
    output_column = 'Hogwarts House'

    dataframe = data_operations().get_data()
    df_min_max_scaled = dataframe.copy()
    dataframe = df_min_max_scaled
    classifiers = k_means_optimiser(iterations=10)
    classifiers.train(dataframe, input_columns, output_column, max_iterations=100)
#   print("The dunn index : %f" % (classifiers.best_classifier.dunn_index()))
    test_data = data_operations(data_source='./datasets/dataset_test.csv').get_data()
    result = classifiers.predict(test_data, input_columns, output_column)
    result.to_csv('output.csv', index=False)
    dataframe = dataframe.dropna(subset=input_columns)
#    print("Accuracy :", accuracy_score(dataframe[output_column], result[output_column]))
#    data_ops = data_operations(data_source='output.csv')
#    draw_scatter_plot(data_ops, *input_columns, fontsize=10)
#    plt.show()
