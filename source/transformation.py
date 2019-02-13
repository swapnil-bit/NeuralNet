import numpy as np
from abc import abstractmethod
from source.activation import Activation


class Transformation:

    @abstractmethod
    def transform(self, input: np.array, weights: np.array) -> np.array:
        pass

    @abstractmethod
    def backpropagate_delta(self, delta_vectors: [np.array], weights: [np.array], activation: Activation,
                            transformed_input: np.array) -> np.array:
        pass


class Linear(Transformation):
    def transform(self, input_array: np.array, weights_array: np.array) -> np.array:
        transformation_axes = self.get_tensordot_axis_length(input_array.shape, weights_array.shape)
        tensor_dot_product = np.tensordot(weights_array, input_array, axes=transformation_axes)
        return tensor_dot_product

    def backpropagate_delta(self, delta_to_layer: np.array, weights_array: np.array, activation: Activation,
                            transformed_input_from_layer: np.array) -> np.array:
        transformation_axes = self.get_tensordot_axis_length(transformed_input_from_layer.shape, weights_array.shape)
        initial_axis = list(np.arange(len(weights_array.shape)))
        transposed_axis = initial_axis[-transformation_axes:] + initial_axis[:-transformation_axes]
        transposed_weights_array = np.transpose(weights_array, transposed_axis)
        backpropagation_axes = len(weights_array.shape) - transformation_axes
        backpropagation_delta = np.tensordot(transposed_weights_array, delta_to_layer, axes=backpropagation_axes)
        # Below is Hadamard product with respect to activation_derivative
        delta_from_layer = backpropagation_delta * activation.derivative(transformed_input_from_layer)
        return delta_from_layer

    def get_tensordot_axis_length(self, input_shape: [], weights_shape: []) -> int:
        loop_length = min(len(input_shape), len(weights_shape))
        tensordot_axis_length = 0
        for i in np.arange(loop_length):
            input_sub_shape = input_shape[:(i + 1)]
            weights_sub_shape = weights_shape[-(i + 1):]
            if input_sub_shape == weights_sub_shape:
                tensordot_axis_length = len(input_sub_shape)
        return min(tensordot_axis_length, len(input_shape), len(weights_shape) - 1)


class Quadratic(Transformation):
    def transform(self, input: np.array, weights: np.array) -> np.array:
        return np.tensordot(input * input, weights.transpose(1, 0), axes=1)

    def backpropagate_delta(self, delta_vectors: [np.array], weights: [np.array], activation: Activation,
                            transformed_input: np.array) -> np.array:
        return 2 * np.tensordot(delta_vectors, weights, axes=1) * activation.function(
            transformed_input) * activation.derivative(transformed_input)
