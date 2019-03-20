import numpy as np

from source.layer import Layer
from source.transformation import Transformation, Linear


class Connection:
    def __init__(self, from_layer: Layer, to_layer: Layer, connection_type: str = "Fully",
                 transformation: Transformation = Linear(), initial_weights: np.array = np.array([])):
        self.id = (from_layer.id, to_layer.id)
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.input_shape = self.from_layer.shape
        self.output_shape = self.to_layer.shape
        self.connection_type = connection_type
        self.transformation = transformation
        self.weights = self.get_weights(initial_weights)

    def get_weights(self, initial_weights) -> [int]:
        if self.connection_type == "Fully":
            weights_shape = self.output_shape + self.input_shape
        else:
            weights_shape = self.get_optimum_weights_shape()
        if list(initial_weights.shape) == weights_shape:
            return initial_weights
        return np.zeros(weights_shape)

    def get_optimum_weights_shape(self):  # Required only for not fully-connected layers
        # if isinstance(self.transformation, Linear):
        max_loop_length = min(len(self.input_shape), len(self.output_shape))
        common_dimension_length = 0
        for i in range(max_loop_length):
            if self.input_shape[-(i + 1)] == self.output_shape[-(i + 1)]:
                common_dimension_length += 1
            else:
                break
        weight_dimension = self.output_shape[:(len(self.output_shape) - common_dimension_length)] + self.input_shape[:(
                    len(self.input_shape) - common_dimension_length)]
        return weight_dimension

    def transform_input(self, input_array) -> np.array:
        return self.transformation.transform(input_array, self.weights)

    def update_weights(self):
        pass
