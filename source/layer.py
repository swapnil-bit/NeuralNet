import numpy as np
from source.activation import Activation, Sigmoid
from source.transformation import Transformation
from source.linear import Linear


class Layer:
    def __init__(self, id: int, shape: [int], name: str = None, initial_bias: np.array = None,
                 activation: Activation = Sigmoid()):
        self.id = id
        self.name = name
        shape = [i for i in shape if i > 1]
        self.shape = [1] if shape == [] else shape
        self.bias = np.zeros(self.shape) if initial_bias is None else initial_bias.reshape(self.shape)
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
        self.input_array = sum(input_arrays).reshape(self.shape) + self.bias

    def set_output_array(self) -> np.array:
        self.output_array = self.activation.function(self.input_array)
