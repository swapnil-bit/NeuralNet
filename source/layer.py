import numpy as np
from source.activation import Activation, Sigmoid


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

    def set_input_array(self, input_arrays: [[np.array]]) -> [np.array]:
        """
        :param  input_arrays: It is a list of lists of arrays. T
                The outermost list takes care of multiple input connections that a layer might be having. The inner list
                takes care of batch sizes.
                Let's say the current layer is of shape (m, n). Let's also say that current layer is  connected to 'p'
                predecessors and we are having inputs in the batch size of 'b'. In this case, len(input_arrays) = p;
                len(input_arrays[0]) = b; and input_arrays[0][0].shape = (m,n)
        :return: After combining inputs from all predecessors, it gives a list of length b (batch size) having arrays
                of shape (m, n)
        """
        # TODO: There can be many ways to combine inputs from multiple predecessors. Currently, it adds only.
        batch_size = len(input_arrays[0])
        combined_input_list = [sum([input_array[i].reshape(self.shape) for input_array in input_arrays]) for i in
                               range(batch_size)]
        self.input_array = [(combined_input + self.bias) for combined_input in combined_input_list]

    def set_output_array(self) -> np.array:
        self.output_array = [self.activation.function(input_array) for input_array in self.input_array]
