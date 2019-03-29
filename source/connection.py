import numpy as np

from source.layer import Layer
from source.transformation import Transformation
from source.linear import Linear


class Connection:
    def __init__(self, input_layer: Layer, output_layer: Layer, connection_type: str = "fully",
                 transformation: Transformation = None, initial_weights: np.array = np.array([]),
                 initial_weights_distribution: str = "zeros"):
        self.id = (input_layer.id, output_layer.id)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.connection_type = connection_type
        self.transformation = transformation
        if self.transformation is None:
            self.transformation = Linear(self.connection_type, self.input_layer.shape, self.output_layer.shape)
        self.weights = self.initialize_weights(initial_weights, initial_weights_distribution)

    def initialize_weights(self, initial_weights, initial_weights_distribution) -> [int]:
        if list(initial_weights.shape) == self.transformation.weights_shape:
            return initial_weights
        if initial_weights_distribution == "normal":
            return np.random.normal(0, 1, self.transformation.weights_shape)
        return np.zeros(self.transformation.weights_shape)

    def transform_input(self, input_array: np.array) -> [np.array]:
        return self.transformation.transform(input_array, self.weights)

    def get_input_layer_delta(self) -> [np.array]:
        return self.transformation.back_propagate_delta(self.output_layer.delta, self.weights,
                                                        self.input_layer.activation, self.input_layer.input_array)

    def get_gradient_for_weights(self) -> np.array:
        return self.transformation.get_gradient_for_weights(self.output_layer.delta, self.input_layer.output_array)

    def update_weights(self):
        pass
