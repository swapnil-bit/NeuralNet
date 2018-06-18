from codeBase.layer import Layer
import numpy as np
import unittest


class LayerClassTests(unittest.TestCase):
    def testThat_setLinearOutput_and_setActivatedOutput_givesCorrectOutputs(self):
        def activation_function(input: np.array) -> np.array:
            return 2*input

        def activation_derivative(input: np.array) -> np.array:
            return np.full(input.shape, 2)

        layer_one = Layer(3, activation_function, activation_derivative)
        input_vectors = np.array([[1, 2, 3, 4], [0, 1, 2, 3]])
        input_weights = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

        layer_one.set_linear_output(input_vectors, input_weights)
        layer_one.set_activated_output()

        expected_linear_output = np.array([[0, 10, 20], [0, 6, 12]], dtype=float)
        expected_activated_output = np.array([[0, 20, 40], [0, 12, 24]], dtype=float)
        self.assertTrue((expected_linear_output == layer_one.linear_output).all())
        self.assertTrue((expected_activated_output == layer_one.activated_output).all())
