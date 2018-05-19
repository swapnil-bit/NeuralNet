import numpy as np


class NetworkConfigurations:
    def cross_entropy_cost_function(self, predicted_y: np.array, actual_y: np.array):
        return np.mean(-(actual_y*np.log(predicted_y) + (1-actual_y)*np.log(1-predicted_y)))

    def get_delta_last_layer(self, predicted_y: np.array, actual_y: np.array):
        # vale is same as cost_derivative * sigmoid_derivative for cross_entropy cost function
        return predicted_y - actual_y

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_derivative(self, z: np.array):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def network_training(self, training_data, max_iteration, eta):
        for iteration in range(max_iteration):
            gradient_for_biases, gradient_for_weights = self.back_propagate_all_layers(training_data)
            self.weights = [w - (eta/len(training_data))*gw for w, gw in zip(self.weights, gradient_for_weights)]
            self.biases = [b - (eta/len(training_data))*gb for b, gb in zip(self.biases, gradient_for_biases)]