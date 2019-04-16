import numpy as np
from abc import abstractmethod


class Loss:
    @abstractmethod
    def function(self, predicted_y: [np.array], actual_y: [np.array]) -> [float]:
        pass

    @abstractmethod
    def get_delta_last_layer(self, predicted_y: [np.array], actual_y: [np.array]) -> [np.array]:
        pass


class CrossEntropy(Loss):
    def function(self, predicted_y: [np.array], actual_y: [np.array]) -> [float]:
        single_y_shape = [1] + list(predicted_y[0].shape)
        predicted_y = [y.reshape(single_y_shape) for y in predicted_y]
        predicted_y = np.concatenate(predicted_y, axis=0)
        actual_y = [y.reshape(single_y_shape) for y in actual_y]
        actual_y = np.concatenate(actual_y, axis=0)
        element_wise_loss = -(actual_y * np.log(predicted_y) + (1 - actual_y) * np.log(1 - predicted_y))
        summary_axis = tuple(range(len(single_y_shape))[1:])
        loss = list(np.mean(element_wise_loss, axis=summary_axis))
        return loss

    def get_delta_last_layer(self, predicted_y: [np.array], actual_y: [np.array]) -> [np.array]:
        # value is same as cost_derivative * sigmoid_derivative for cross_entropy cost function
        actual_y = [y.reshape(predicted_y[0].shape) for y in actual_y]
        last_layer_delta = [predicted_y[i] - actual_y[i] for i in range(len(predicted_y))]
        return last_layer_delta


class MeanSquare(Loss):
    def function(self, predicted_y: [np.array], actual_y: [np.array]) -> [float]:
        actual_y = [y.reshape(predicted_y[0].shape) for y in actual_y]
        loss = [np.mean((actual_y[i] - predicted_y[i]) ** 2) for i in range(len(predicted_y))]
        return loss

    def get_delta_last_layer(self, predicted_y: [np.array], actual_y: [np.array]) -> [np.array]:
        # TODO: To be corrected. Currently written same as CrossEntropy.
        actual_y = [y.reshape(predicted_y[0].shape) for y in actual_y]
        last_layer_delta = [predicted_y[i] - actual_y[i] for i in range(len(predicted_y))]
        return last_layer_delta
