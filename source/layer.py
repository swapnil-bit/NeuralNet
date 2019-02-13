import numpy as np
from source.activation import Activation, Sigmoid
from source.transformation import Transformation, Linear


class Layer:
    def __init__(self, id: int, shape: [int], name: str = None, initial_bias: np.array = None,
                 input_transformation: Transformation = Linear(), activation: Activation = Sigmoid()):
        self.id = id
        self.name = name
        self.shape = shape
        self.bias = np.zeros(shape) if initial_bias is None else initial_bias
        self.input_transformation = input_transformation
        self.activation = activation
        self.predecessors = []
        self.successors = []
        self.input_array = np.zeros(self.shape, dtype=float)
        self.output_array = np.zeros(self.shape, dtype=float)
        self.delta = np.zeros(self.shape, dtype=float)

    def set_predecessor_list(self, predecessors: [int]):
        self.predecessors = predecessors

    def set_successor_list(self, successors: [int]):
        self.successors = successors

    def set_input_array(self, input_arrays: [np.array]) -> np.array:
        # TODO: How to combine inputs from multiple predecessors? As first step, it will add only.
        self.input_array = sum(input_arrays) + self.bias

    def set_output_array(self) -> np.array:
        self.output_array = self.activation.function(self.input_array)

    def set_delta(self, delta_vectors: [np.array], weights: [np.array]):
        self.delta = self.input_transformation.backpropagate_delta(delta_vectors, weights, self.activation, self.input_array)
        # self.delta = np.tensordot(delta_vectors, weights, axes=1) * self.activation.derivative(self.transformed_input)
        # TODO: First part above will change as per input_transformation -> needs to be moved elsewhere as transformation is not part of layer
