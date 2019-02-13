from source.layer import Layer
from source.activation import LinearDouble
import numpy as np
import unittest


class LayerClassTests(unittest.TestCase):
    def testThat_setInputArray_and_setOutputArray_givesCorrectOutputs_for1DLayerShape(self):
        layer_one = Layer(0, [2], activation=LinearDouble())
        input_arrays = [np.array([1, 2]), np.array([3, 4]), np.array([0, 1]), np.array([2, 3])]

        expected_input_array = np.array([6, 10], dtype=float)
        expected_output_array = np.array([12, 20], dtype=float)

        layer_one.set_input_array(input_arrays)
        layer_one.set_output_array()

        self.assertTrue((expected_input_array == layer_one.input_array).all())
        self.assertTrue((expected_output_array == layer_one.output_array).all())

    def testThat_setInputArray_and_setOutputArray_givesCorrectOutputs_for2DLayerShape(self):
        layer_one = Layer(0, [2, 3], activation=LinearDouble())
        input_arrays = [np.array([[1, 2, 3], [4, 0, 1]])]

        expected_input_array = np.array([[1, 2, 3], [4, 0, 1]], dtype=float)
        expected_output_array = np.array([[2, 4, 6], [8, 0, 2]], dtype=float)

        layer_one.set_input_array(input_arrays)
        layer_one.set_output_array()

        self.assertTrue((expected_input_array == layer_one.input_array).all())
        self.assertTrue((expected_output_array == layer_one.output_array).all())
