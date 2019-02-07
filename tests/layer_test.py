from source.layer import Layer
from source.activation import LinearDouble
from source.transformation import Linear, Quadratic
import numpy as np
import unittest


class LayerClassTests(unittest.TestCase):
    def testThat_setTransformedInput_and_setActivatedOutput_givesCorrectOutputs_WithLinearTransformation(self):
        layer_one = Layer([3], input_transformation=Linear(), activation=LinearDouble())
        input_vectors = np.array([[1, 2, 3, 4], [0, 1, 2, 3]])
        input_weights = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

        layer_one.set_transformed_input(input_vectors, input_weights)
        layer_one.set_activated_output()

        expected_transformed_input = np.array([[0, 10, 20], [0, 6, 12]], dtype=float)
        expected_activated_output = np.array([[0, 20, 40], [0, 12, 24]], dtype=float)
        self.assertTrue((expected_transformed_input == layer_one.transformed_input).all())
        self.assertTrue((expected_activated_output == layer_one.activated_output).all())

    def testThat_setTransformedInput_and_setActivatedOutput_givesCorrectOutputs_WithQuadraticTransformation(self):
        layer_one = Layer([3], input_transformation=Quadratic(), activation=LinearDouble())
        input_vectors = np.array([[1, 2, 3, 4], [0, 1, 2, 3]])
        input_weights = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

        layer_one.set_transformed_input(input_vectors, input_weights)
        layer_one.set_activated_output()

        expected_transformed_input = np.array([[0, 30, 60], [0, 14, 28]], dtype=float)
        expected_activated_output = np.array([[0, 60, 120], [0, 28, 56]], dtype=float)
        self.assertTrue((expected_transformed_input == layer_one.transformed_input).all())
        self.assertTrue((expected_activated_output == layer_one.activated_output).all())

    # def testThat_setTransformedInput_and_setActivatedOutput_givesCorrectOutputs_With2DShape(self):
    #     layer_one = Layer([3, 2], input_transformation=Linear(), activation=LinearDouble())
    #     input_vectors = np.array([[1, 2, 3, 4], [0, 1, 2, 3]])
    #     input_weights = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
    #
    #     layer_one.set_transformed_input(input_vectors, input_weights)
    #     layer_one.set_activated_output()
    #
    #     expected_transformed_input = np.array([[0, 10, 20], [0, 6, 12]], dtype=float)
    #     expected_activated_output = np.array([[0, 20, 40], [0, 12, 24]], dtype=float)
    #     self.assertTrue((expected_transformed_input == layer_one.transformed_input).all())
    #     self.assertTrue((expected_activated_output == layer_one.activated_output).all())
