import numpy as np
from abc import abstractmethod
from source.activation import Activation


class Transformation:

    @abstractmethod
    def transform(self, input: np.array, weights: np.array) -> np.array:
        pass

    @abstractmethod
    def back_propagate_delta(self, delta_vectors: [np.array], weights: [np.array], activation: Activation,
                             transformed_input: np.array) -> np.array:
        pass

    @abstractmethod
    def get_gradient_for_weights(self, successor_delta: np.array, predecessor_output: np.array) -> np.array:
        pass


class Quadratic(Transformation):
    def transform(self, input: np.array, weights: np.array) -> np.array:
        return np.tensordot(input * input, weights.transpose(1, 0), axes=1)

    def back_propagate_delta(self, delta_vectors: [np.array], weights: [np.array], activation: Activation,
                             transformed_input: np.array) -> np.array:
        return 2 * np.tensordot(delta_vectors, weights, axes=1) * activation.function(
            transformed_input) * activation.derivative(transformed_input)

    def get_gradient_for_weights(self, successor_delta: np.array, predecessor_output: np.array) -> np.array:
        pass
