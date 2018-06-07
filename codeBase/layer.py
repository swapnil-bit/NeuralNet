import numpy as np
from typing import Callable
from codeBase.networkConfigurations import NetworkConfigurations


class Layer:
    def __init__(self, size: int, activation_function: Callable[[np.array], np.array], activation_derivative: Callable[[np.array], np.array]):
        self.size = size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.predecessors = []
        self.successors = []
        self.bias = np.zeros([self.size, 1], dtype = float)
        self.linear_output = np.zeros([self.size, 1], dtype = float)
        self.activated_output = np.zeros([self.size, 1], dtype = float)
        self.delta = np.zeros([self.size, 1], dtype = float)
        self.config = NetworkConfigurations()

    def set_predecessor_list(self, predecessors: [int]):
        self.predecessors = predecessors

    def set_successor_list(self, successors: [int]):
        self.successors = successors

    def set_linear_output(self, input_vectors: [np.array], weights: [np.array]) -> np.array:
        all_weights = np.concatenate(weights, axis = 1)
        all_inputs = np.concatenate(input_vectors, axis = 0)
        self.linear_output = np.dot(all_weights, all_inputs) + self.bias

    def set_activated_output(self) -> np.array:
        self.activated_output = self.activation_function(self.linear_output)

    def set_delta(self, delta_vectors: [np.array], weights: [np.array]):
        all_weights = np.concatenate(weights, axis = 0)
        all_deltas = np.concatenate(delta_vectors, axis = 0)
        self.delta = np.dot(all_weights.transpose(), all_deltas) * self.activation_derivative(self.linear_output)
