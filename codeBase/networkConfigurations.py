import numpy as np


class NetworkConfigurations:
    # def cross_entropy_cost_function(self, predicted_y: np.array, actual_y: np.array):
    #     return np.mean(-(actual_y*np.log(predicted_y) + (1-actual_y)*np.log(1-predicted_y)))
    #
    # def get_delta_last_layer(self, predicted_y: np.array, actual_y: np.array):
    #     # value is same as cost_derivative * sigmoid_derivative for cross_entropy cost function
    #     return predicted_y - actual_y

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_derivative(self, z: np.array):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def cross_entropy_cost_function(self, predicted_y: np.array, actual_y: np.array):
        return np.mean((actual_y - predicted_y)^2)

    def get_delta_last_layer(self, predicted_y: np.array, actual_y: np.array):
        # value is same as cost_derivative * sigmoid_derivative for cross_entropy cost function
        return (predicted_y - actual_y)
