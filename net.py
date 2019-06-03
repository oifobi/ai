from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class NetModel:

    @staticmethod
    def activation(s: np.matrix):
        return np.divide(1, 1 + np.exp(-s))

    @staticmethod
    def error(y: np.matrix, expected: np.matrix):
        return -np.log(1 - expected + np.multiply(y, 2 * expected - 1))

    @staticmethod
    def df_ds(s: np.matrix):
        y = NetModel.activation(s)
        return np.multiply(y, 1 - y)

    @staticmethod
    def dE_dy(y: np.matrix, expected: np.matrix):
        return np.divide(1, 1 - expected - y)

    @staticmethod
    def build_net(feature_vector_size: int,
                  hidden_layer_widths: [int],
                  output_vector_size: int,
                  learning_rate: float):
        return Net(feature_vector_size,
                   hidden_layer_widths,
                   output_vector_size,
                   learning_rate,
                   NetModel.activation,
                   NetModel.df_ds,
                   NetModel.dE_dy)


class Net:

    # Symbols:
    # f:s->y - activation function
    # s - sum
    # y - output
    # E - error
    # w - weight
    # d?_d# - derivative of ? with respect to #
    # m - learning_rate (should be 0<m<1 and it makes sense to keep it very close to 0)
    def __init__(self,
                 feature_vector_size: int,
                 hidden_layer_widths: [int],
                 output_vector_size: int,
                 learning_rate: float,
                 activation_func,
                 derivative_func,
                 err_derivative_func):
        self.weight_matrices: [np.matrix] = []
        previous_width: int = feature_vector_size
        for width in hidden_layer_widths + [output_vector_size]:
            self.weight_matrices.append(np.random.rand(previous_width + 1, width))
            previous_width = width
        self.f = activation_func
        self.df_ds = derivative_func
        self.dE_dy = err_derivative_func
        self.learning_rate = learning_rate

    def train_rec(self, input_matrix: np.matrix, expected_matrix: np.matrix, layer: int = 0) -> np.matrix:
        if layer == len(self.weight_matrices):
            return self.dE_dy(input_matrix, expected_matrix)
        weight_matrix: np.matrix = self.weight_matrices[layer]
        bias = np.ones((len(input_matrix), 1))
        bias_and_input: np.matrix = np.hstack((input_matrix, bias))
        sum_matrix: np.matrix = bias_and_input @ weight_matrix
        dE_dX_matrix: np.matrix = self.train_rec(self.f(sum_matrix), expected_matrix, layer + 1)
        dX_dS_matrix: np.matrix = self.df_ds(sum_matrix)
        dE_dS_matrix: np.matrix = np.multiply(dE_dX_matrix, dX_dS_matrix)
        dE_dW_matrix = bias_and_input.T @ dE_dS_matrix
        dE_dprevX = (weight_matrix @ dE_dS_matrix.T).T
        self.weight_matrices[layer] -= dE_dW_matrix * self.learning_rate
        return np.matrix(np.delete(dE_dprevX, -1, axis=1))

    def propagate(self, input_matrix: np.matrix) -> np.matrix:
        bias_vector: np.matrix = np.ones((len(input_matrix), 1))
        for weight_matrix in self.weight_matrices:
            bias_and_input: np.matrix = np.hstack((input_matrix, bias_vector))
            input_matrix: np.matrix = self.f(bias_and_input @ weight_matrix)
        return input_matrix

    def plot_mesh(self,
                  input_matrix: np.matrix,
                  expected_matrix: np.matrix,
                  left: float = None, right: float = None,
                  top: float = None, bottom: float = None,
                  step: float = None):
        if left is None:
            left = np.min(input_matrix[:, 0])
        if right is None:
            right = np.max(input_matrix[:, 0])
        if bottom is None:
            bottom = np.min(input_matrix[:, 1])
        if top is None:
            top = np.max(input_matrix[:, 1])
        if step is None:
            step = (right - left) / 25
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(left, right, step)
        Y = np.arange(bottom, top, step)
        X, Y = np.meshgrid(X, Y)

        flatX = np.matrix(X).flatten()
        flatY = np.matrix(Y).flatten()

        stacked = np.matrix(np.hstack((flatX.T, flatY.T)))
        result = self.propagate(stacked)
        Z = np.reshape(result, np.matrix(Y).shape)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, np.sin(Z), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.scatter(input_matrix[:, 0], input_matrix[:, 1], expected_matrix, c='g', marker='o')
        plt.show()

    def iterate(self,
                input_matrix: np.matrix,
                expected_matrix: np.matrix,
                iterations: int):
        def progress(count, total, status=''):
            bar_len = 60
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)

            sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
            sys.stdout.flush()

        for i in range(iterations):
            progress(i, iterations, status='Training')
            self.train_rec(input_matrix, expected_matrix)
        print()

    def iterate_and_show(self,
                         input_matrix: np.matrix,
                         expected_matrix: np.matrix,
                         iterations: int,
                         left: float = None, right: float = None,
                         top: float = None, bottom: float = None,
                         step: float = None):
        self.iterate(input_matrix, expected_matrix, iterations)
        self.plot_mesh(input_matrix, expected_matrix, left, right, top, bottom, step)

    def print_accuracy(self, input_matrix: np.matrix, expected_matrix: np.matrix) -> float:
        result = self.propagate(input_matrix)
        error = abs(expected_matrix - result)
        error_sum = float(sum(error))
        normalised_error_sum = error_sum/len(input_matrix)
        print("Error absolute: " + str(error_sum))
        print("Error normalised: "+str(normalised_error_sum)+" (len="+str(len(input_matrix))+")")
        return normalised_error_sum



