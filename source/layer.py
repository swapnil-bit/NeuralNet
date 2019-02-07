import numpy as np
from source.activation import Activation, Sigmoid
from source.transformation import Transformation, Linear


class Layer:
    def __init__(self, shape: [int], initial_bias: np.array = None,
                 input_transformation: Transformation = Linear(), activation: Activation = Sigmoid()):
        self.shape = shape
        self.bias = np.zeros(shape) if initial_bias is None else initial_bias
        self.input_transformation = input_transformation
        self.activation = activation
        self.predecessors = []
        self.successors = []
        self.transformed_input = np.zeros(self.shape, dtype=float)
        self.activated_output = np.zeros(self.shape, dtype=float)
        self.delta = np.zeros(self.shape, dtype=float)

    def set_predecessor_list(self, predecessors: [int]):
        self.predecessors = predecessors

    def set_successor_list(self, successors: [int]):
        self.successors = successors

    def set_transformed_input(self, input_vectors: np.array, weights: np.array) -> np.array:
        self.transformed_input = self.input_transformation.transform(input_vectors, weights) + self.bias

    def set_activated_output(self) -> np.array:
        self.activated_output = self.activation.function(self.transformed_input)

    def set_delta(self, delta_vectors: [np.array], weights: [np.array]):
        self.delta = self.input_transformation.backpropagate_delta(delta_vectors, weights, self.activation, self.transformed_input)
        # self.delta = np.tensordot(delta_vectors, weights, axes=1) * self.activation.derivative(self.transformed_input)
        # TODO: First part above will change as per input_transformation
