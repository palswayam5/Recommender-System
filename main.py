import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class Learning_Curves():

    def __init__(self, X, y, no_of_examples):
        self.X = X
        self.y = y
        self.size_of_x = no_of_examples

        self.Linear_regression_error = np.zeros(4)
        self.cross_validation_error = np.zeros(4)
        self.index = 0
        self.connections()

    def set_size(self):
        self.train_set_size = [0.2, 0.3, 0.4, 0.5]
        self.i = self.train_set_size[self.index]

    def define_train_cross_validation_sets(self):
        self.train_set_X = self.X[0:int(self.i*self.size_of_x)]
        self.train_set_y = self.y[0:int(self.i*self.size_of_x)].reshape(-1, 1)
        self.cross_validation_size = 1 - self.i
        self.cross_validation_set_X = self.X[int(self.i*self.size_of_x):int(
            self.i*self.size_of_x)+int(self.size_of_x*(self.cross_validation_size))]
        self.cross_validation_set_y = self.y[int(self.i*self.size_of_x):int(
            self.i*self.size_of_x)+int(self.size_of_x*(self.cross_validation_size))].reshape(-1, 1)

    def training_both_sets(self):
        self.training_set_fit = LinearRegression().fit(
            self.train_set_X, self.train_set_y)
        self.cross_validation_set_fit = LinearRegression().fit(
            self.cross_validation_set_X, self.cross_validation_set_y)

    def calculate_error(self):
        self.Linear_regression_error[self.index] = (
            self.training_set_fit.score(self.train_set_X, self.train_set_y))
        self.cross_validation_error[self.index] = (self.cross_validation_set_fit.score(
            self.cross_validation_set_X, self.cross_validation_set_y))

    def plot_curve(self):
        plt.plot(self.train_set_size,
                 self.Linear_regression_error, 'r')
        plt.plot(self.train_set_size,
                 self.cross_validation_error, 'b')
        plt.xlabel('Training set size')
        plt.ylabel('Error')
        plt.show()

    def connections(self):
        if (self.index < 4):
            self.set_size()
            self.define_train_cross_validation_sets()
            self.training_both_sets()
            self.calculate_error()
            self.index += 1
            self.connections()
        else:
            self.plot_curve()


# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
# y = np.array([1, 2, 3, 4, 5])

# taylor_swift = Learning_Curves(X, y, 5)
