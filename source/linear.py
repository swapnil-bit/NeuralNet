import numpy as np

from source.activation import Activation
from source.transformation import Transformation


class Linear(Transformation):
    def __init__(self, connection_type: str, input_shape: [int], output_shape: [int]):
        self.connection_type = connection_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights_shape, self.forward_propagation_axes = self.get_forward_propagation_parameters()
        self.transposed_weights_axes, self.back_propagation_axes = self.get_back_propagation_parameters()
        self.transposed_input_axes, self.weight_gradient_axes = self.get_gradient_parameters()

    def get_forward_propagation_parameters(self) -> ([int], int):
        if self.connection_type == "fully":
            weights_shape = self.output_shape + self.input_shape
            transformation_axis = len(self.input_shape)
            return weights_shape, transformation_axis

        # TODO: try finding a better way for below code piece
        max_loop_length = min(len(self.input_shape), len(self.output_shape))
        common_dimension_length = 0
        for i in range(max_loop_length):
            if self.input_shape[-(i + 1)] == self.output_shape[-(i + 1)]:
                common_dimension_length += 1
            else:
                break
        weights_shape = self.output_shape[:(len(self.output_shape) - common_dimension_length)] \
                        + self.input_shape[:(len(self.input_shape) - common_dimension_length)]

        if len(weights_shape) == 0:
            return [1], 0

        return weights_shape, (len(self.input_shape) - common_dimension_length)

    def get_back_propagation_parameters(self) -> ([int], int):
        if self.weights_shape == [1]:
            return [0], 0
        back_propagation_axes = len(self.weights_shape) - self.forward_propagation_axes
        weight_axes = list(np.arange(len(self.weights_shape)))
        transposed_weight_axes = weight_axes[back_propagation_axes:] + weight_axes[:back_propagation_axes]
        return transposed_weight_axes, back_propagation_axes

    def get_gradient_parameters(self) -> ([int], int):
        gradient_axes = len(self.input_shape) - self.forward_propagation_axes
        input_axes = list(np.arange(len(self.input_shape)))
        transposed_input_axes = input_axes[self.forward_propagation_axes:] + input_axes[:self.forward_propagation_axes]
        return transposed_input_axes, gradient_axes

    def transform(self, input_array: np.array, weights: np.array) -> np.array:
        tensor_dot_product = np.tensordot(weights, input_array, axes=self.forward_propagation_axes)
        return tensor_dot_product

    def back_propagate_delta(self, output_layer_delta: np.array, weights: np.array, activation: Activation,
                             transformed_input: np.array) -> np.array:
        transposed_weights = np.transpose(weights, self.transposed_weights_axes)
        back_propagation_delta = np.tensordot(transposed_weights, output_layer_delta, axes=self.back_propagation_axes)
        if list(weights.shape) == [1]:
            back_propagation_delta = np.squeeze(back_propagation_delta, axis=0)
        input_layer_delta = back_propagation_delta * activation.derivative(transformed_input)  # Hadamard Product
        return input_layer_delta

    def get_gradient_for_weights(self, output_layer_delta: np.array, activated_input: np.array) -> np.array:
        transposed_predecessor_output = np.transpose(activated_input, self.transposed_input_axes)
        gradient_for_weights = np.tensordot(output_layer_delta, transposed_predecessor_output,
                                            axes=self.weight_gradient_axes)
        return gradient_for_weights
