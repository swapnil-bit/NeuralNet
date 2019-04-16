import numpy as np

from source.activation import Activation
from source.transformation import Transformation


class Convolution(Transformation):
    def __init__(self):
        pass

    def transform(self, input_array: [np.array], weights: np.array) -> [np.array]:
        pass

    def back_propagate_delta(self, output_layer_delta: [np.array], weights: np.array, activation: Activation,
                             transformed_input: [np.array]) -> [np.array]:
        pass

    def get_gradient_for_weights(self, output_layer_delta: [np.array], activated_input: [np.array]) -> np.array:
        pass
