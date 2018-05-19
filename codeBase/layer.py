import numpy as np
from codeBase.networkConfigurations import NetworkConfigurations

class Layer:
    def __init__(self, size: int, predecessor_list: [int], successor_list: [int]):
        self.size = size
        self.predecessors = predecessor_list
        self.successors = successor_list
        self.bias = np.zeros([self.size, 1], dtype = float)
        self.linear_output = np.zeros([self.size, 1], dtype = float)
        self.activated_output = np.zeros([self.size, 1], dtype = float)
        self.delta = np.zeros([self.size, 1], dtype = float)
        self.config = NetworkConfigurations()

    def set_linear_output(self, input_vectors: [np.array], weights: [np.array]) -> np.array:
        all_weights = np.concatenate(weights, axis = 1)
        all_inputs = np.concatenate(input_vectors, axis = 0)
        self.linear_output = np.dot(all_weights, all_inputs) + self.bias

    def set_activated_output(self) -> np.array:
        self.activated_output = 1.0/(1.0 + np.exp(-self.linear_output))

    def set_delta(self, delta_vectors: [np.array], weights: [np.array]):
        all_weights = np.concatenate(weights, axis = 0)
        all_deltas = np.concatenate(delta_vectors, axis = 0)
        self.delta = np.dot(all_weights.transpose(), all_deltas) * self.config.sigmoid_derivative(self.linear_output)
