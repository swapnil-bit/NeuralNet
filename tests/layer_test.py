from codeBase.layer import Layer
import numpy as np
import unittest


class LayerClassTests(unittest.TestCase):
    def testThat_setLinearOutput_and_setActivatedOutput_givesCorrectOutputs(self):
        def activation_function(input: np.array) -> np.array:
            return 2*input

        def activation_derivative(input: np.array) -> np.array:
            return np.full(input.shape, 2)

        layerOne = Layer(3, activation_function, activation_derivative)
        inputVectors = [np.array([1, 2, 3, 4])]
        inputWeights = [np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])]
        layerOne.set_linear_output(inputVectors, inputWeights)
        layerOne.set_activated_output()

        expected_linear_output = np.array([0, 10, 20])
        expected_activated_output = np.array([0, 20, 40])
        self.assertTrue((expected_linear_output == layerOne.linear_output).all())
        self.assertTrue((expected_activated_output == layerOne.activated_output).all())
