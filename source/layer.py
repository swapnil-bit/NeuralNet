import numpy as np
from source.activation import Activation, Sigmoid
from source.transformation import Transformation
from source.linear import Linear


class Layer:
    # TODO: need to investigate for 1D layers, is [n] correct dimension or [1, n]? Currently, it works with [n]
    def __init__(self, id: int, shape: [int], name: str = None, initial_bias: np.array = None,
                 activation: Activation = Sigmoid()):
        self.id = id
        self.name = name
        self.shape = shape
        self.bias = np.zeros(shape) if initial_bias is None else initial_bias
        # self.input_transformation = input_transformation
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
