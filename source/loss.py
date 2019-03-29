import numpy as np
from abc import abstractmethod


class Loss:
    @abstractmethod
    def function(self, predicted_y: np.array, actual_y: np.array) -> np.array:
        pass

    @abstractmethod
    def get_delta_last_layer(self, predicted_y: [np.array], actual_y: [np.array]) -> [np.array]:
        pass


class CrossEntropy(Loss):
    def function(self, predicted_y: np.array, actual_y: np.array):
        # TODO: Check if this needs to be a list input and list output
        return np.mean(-(actual_y * np.log(predicted_y) + (1 - actual_y) * np.log(1 - predicted_y)))

    def get_delta_last_layer(self, predicted_y: [np.array], actual_y: [np.array]) -> [np.array]:
        # value is same as cost_derivative * sigmoid_derivative for cross_entropy cost function
        actual_y = [y.reshape(predicted_y[0].shape) for y in actual_y]
        last_layer_delta = [predicted_y[i] - actual_y[i] for i in range(len(predicted_y))]
        return last_layer_delta


class MeanSquare(Loss):
    def function(self, predicted_y: np.array, actual_y: np.array):
        return np.mean((actual_y - predicted_y) ** 2)

    def get_delta_last_layer(self, predicted_y: np.array, actual_y: np.array):
        # value is same as cost_derivative * sigmoid_derivative for cross_entropy cost function
        last_layer_delta = [predicted_y[i] - actual_y[i] for i in range(len(predicted_y))]
        return last_layer_delta
