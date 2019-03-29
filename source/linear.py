import numpy as np

from source.activation import Activation
from source.transformation import Transformation


class Linear(Transformation):
    def __init__(self, connection_type: str, input_layer_shape: [int], output_layer_shape: [int]):
        self.connection_type = connection_type
        input_layer_shape = [i for i in input_layer_shape if i > 1]
        self.input_layer_shape = [1] if input_layer_shape == [] else input_layer_shape
        output_layer_shape = [i for i in output_layer_shape if i > 1]
        self.output_layer_shape = [1] if output_layer_shape == [] else output_layer_shape
        self.weights_shape, self.forward_propagation_axes = self.get_forward_propagation_parameters()
        self.transposed_weights_axes, self.back_propagation_axes = self.get_back_propagation_parameters()
        self.transposed_input_axes, self.weight_gradient_axes = self.get_gradient_parameters()

    def get_forward_propagation_parameters(self) -> ([int], int):
        if self.connection_type == "fully":
            weights_shape = self.output_layer_shape + self.input_layer_shape
            transformation_axis = len(self.input_layer_shape)
            return weights_shape, transformation_axis

        # TODO: try finding a better way for below code piece
        max_loop_length = min(len(self.input_layer_shape), len(self.output_layer_shape))
        common_dimension_length = 0
        for i in range(max_loop_length):
            if self.input_layer_shape[-(i + 1)] == self.output_layer_shape[-(i + 1)]:
                common_dimension_length += 1
            else:
                break
        weights_shape = self.output_layer_shape[:(len(self.output_layer_shape) - common_dimension_length)] \
                        + self.input_layer_shape[:(len(self.input_layer_shape) - common_dimension_length)]

        if len(weights_shape) == 0:
            return [1], 0

        return weights_shape, (len(self.input_layer_shape) - common_dimension_length)

    def get_back_propagation_parameters(self) -> ([int], int):
        if self.weights_shape == [1]:
            return [0], 0
        back_propagation_axes = len(self.weights_shape) - self.forward_propagation_axes
        weight_axes = list(np.arange(len(self.weights_shape)))
        transposed_weight_axes = weight_axes[back_propagation_axes:] + weight_axes[:back_propagation_axes]
        return transposed_weight_axes, back_propagation_axes

    def get_gradient_parameters(self) -> ([int], int):
        gradient_axes = len(self.input_layer_shape) - self.forward_propagation_axes
        input_axes = list(np.arange(len(self.input_layer_shape)))
        transposed_input_axes = input_axes[self.forward_propagation_axes:] + input_axes[:self.forward_propagation_axes]
        return transposed_input_axes, gradient_axes

    def transform(self, input_array: [np.array], weights: np.array) -> [np.array]:
        """
        :param input_array: It is list of b elements where b is the batch size being processed together. Every element
        here in itself is the input to current layer from one of its predecessor in the form of an array. So, it is same
        as the output array of the predecessor. The shape of a single input array would be mostly same as
        self.input_shape. If not, then it should be reshaped to make it so.
        :param weights: Weights is a tensor defining relationship between input and output layer of this connection.
        :return: Returns a list of arrays. The length of list is same as b, the batch size being processed together.
        Every element of the list is an array having shape of the output layer.

        LOGIC: Following piece of code can be used instead of actual code, if parallel processing is implemented in some
        other ways:
            input_array = [single_input.reshape(self.input_shape) for single_input in input_array]
            transformed_input = [np.tensordot(weights, single_input, axes=self.forward_propagation_axes) for single_input
                            in input_array]
            return transformed_input
        """
        batch_size = len(input_array)
        input_array = np.concatenate(input_array, axis=0).reshape([batch_size] + list(self.input_layer_shape))
        input_array_axes = list(np.arange(len(input_array.shape)))
        transposed_input_array = np.transpose(input_array, axes=(input_array_axes[1:] + input_array_axes[:1]))
        dot_product = np.tensordot(weights, transposed_input_array, axes=self.forward_propagation_axes)
        dot_product_axes = list(np.arange(len(dot_product.shape)))
        transformed_input = np.transpose(dot_product, axes=(dot_product_axes[-1:] + dot_product_axes[:-1]))
        transformed_input = list(transformed_input)
        return transformed_input

    def back_propagate_delta(self, output_layer_delta: [np.array], weights: np.array, activation: Activation,
                             transformed_input: [np.array]) -> [np.array]:
        # TODO: Logic can be similar to transform function for parallel processing
        output_layer_delta = [single_delta.reshape(self.output_layer_shape) for single_delta in output_layer_delta]
        transformed_input = [single_input.reshape(self.input_layer_shape) for single_input in transformed_input]
        transposed_weights = np.transpose(weights, self.transposed_weights_axes)
        back_propagation_delta = [np.tensordot(transposed_weights, single_delta, axes=self.back_propagation_axes) for
                                  single_delta in output_layer_delta]
        if list(weights.shape) == [1]:
            back_propagation_delta = [np.squeeze(single_delta, axis=0) for single_delta in back_propagation_delta]

        input_layer_delta = [back_propagation_delta[i] * activation.derivative(transformed_input[i]) for i in
                             range(len(transformed_input))]  # Hadamard Product
        return input_layer_delta

    def get_gradient_for_weights(self, output_layer_delta: [np.array], activated_input: [np.array]) -> np.array:
        # TODO: Logic can be similar to transform function for parallel processing
        transposed_predecessor_output = [np.transpose(single_activated_input, self.transposed_input_axes) for
                                         single_activated_input in activated_input]
        gradient_for_weights = [np.tensordot(output_layer_delta[i], transposed_predecessor_output[i],
                                             axes=self.weight_gradient_axes) for i in range(len(output_layer_delta))]
        return sum(gradient_for_weights)
