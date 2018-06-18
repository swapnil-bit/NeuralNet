import numpy as np
from typing import Callable


class Layer:
    def __init__(self, size: int, activation_function: Callable[[np.array], np.array],
                 activation_derivative: Callable[[np.array], np.array]):
        self.size = size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.predecessors = []
        self.successors = []
        self.bias = np.zeros([1, self.size], dtype=float)
        self.linear_output = np.zeros(self.size, dtype=float)
        self.activated_output = np.zeros(self.size, dtype=float)
        self.delta = np.zeros([1, self.size], dtype=float)

    def set_predecessor_list(self, predecessors: [int]):
        self.predecessors = predecessors

    def set_successor_list(self, successors: [int]):
        self.successors = successors

    def set_linear_output(self, input_vectors: np.array, weights: np.array) -> np.array:
        self.linear_output = np.tensordot(input_vectors, weights.transpose(1, 0), axes = 1)

    def set_activated_output(self) -> np.array:
        self.activated_output = self.activation_function(self.linear_output)

    def set_delta(self, delta_vectors: [np.array], weights: [np.array]):
        self.delta = np.tensordot(delta_vectors, weights, axes = 1) * self.activation_derivative(self.linear_output)
