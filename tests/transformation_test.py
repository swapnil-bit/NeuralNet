from source.transformation import Linear
from source.activation import Sigmoid
import numpy as np
import unittest


class TransformationClassTests(unittest.TestCase):
    def testThat_LinearTransformation_givesCorrectAxisLength_with1DLayers(self):
        # Example: from_layer.shape = [10,], to_layer.shape = [10,] => weights.shape = [10,10]; axis_length = 1
        input_shape = [10, ]
        weights_shape = [10, 10]
        transformation1 = Linear()
        actual_axis_length = transformation1.get_tensordot_axis_length(input_shape, weights_shape)
        expected_axis_length = 1
        self.assertEqual(expected_axis_length, actual_axis_length)

    def testThat_LinearTransformation_givesCorrectAxisLength_with1DFromLayerAnd2DToLayer(self):
        # Example: from_layer.shape = [10,], to_layer.shape = [5,10] => weights.shape = [5,10,10]; axis_length = 1
        input_shape = [10, ]
        weights_shape = [5, 10, 10]
        transformation1 = Linear()
        actual_axis_length = transformation1.get_tensordot_axis_length(input_shape, weights_shape)
        expected_axis_length = 1
        self.assertEqual(expected_axis_length, actual_axis_length)

    def testThat_LinearTransformation_givesCorrectAxisLength_with2DFromLayerAnd3DToLayerButFullCommonality(self):
        # Example: from_layer.shape = [10,5], to_layer.shape = [5,10,5] => weights.shape = [5,]; axis_length = 0
        input_shape = [10, 5]
        weights_shape = [5, ]
        transformation1 = Linear()
        actual_axis_length = transformation1.get_tensordot_axis_length(input_shape, weights_shape)
        expected_axis_length = 0
        self.assertEqual(expected_axis_length, actual_axis_length)

    def testThat_LinearTransformation_givesCorrectAxisLength_with2DFromLayerAnd3DToLayerButNoCommonality(self):
        # Example: from_layer.shape = [10,5], to_layer.shape = [2,3,4] => weights.shape = [2,3,4,10,5]; axis_length = 2
        input_shape = [10, 5]
        weights_shape = [2, 3, 4, 10, 5]
        transformation1 = Linear()
        actual_axis_length = transformation1.get_tensordot_axis_length(input_shape, weights_shape)
        expected_axis_length = 2
        self.assertEqual(expected_axis_length, actual_axis_length)

    def testThat_LinearTransformation_givesCorrectAxisLength_with3DFromLayerAnd3DToLayerButPartialCommonality(self):
        # Example: from_layer.shape = [10,5,4], to_layer.shape = [2,3,4] => weights.shape = [2,3,10,5]; axis_length = 2
        input_shape = [10, 5, 4]
        weights_shape = [2, 3, 10, 5]
        transformation1 = Linear()
        actual_axis_length = transformation1.get_tensordot_axis_length(input_shape, weights_shape)
        expected_axis_length = 2
        self.assertEqual(expected_axis_length, actual_axis_length)

    def testThat_LinearTransformation_givesCorrectTransformation_with1DFromLayersAnd2DToLayer(self):
        # Example: from_layer.shape = [3,], to_layer.shape = [3,2] => weights.shape = [3,2,3]; axis_length = 1
        input_array = np.array([1, 0, 1])
        weight_array = np.array([[[2, 1, 0], [2, 2, 3]],
                                 [[1, 1, 0], [1, 2, 0]],
                                 [[0, 1, 2], [1, 1, 1]]])
        transformation2 = Linear()
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([[2, 5], [1, 1], [2, 2]])
        self.assertTrue((actual_output == expected_output).all())

    def testThat_LinearTransformation_givesCorrectTransformation_with2DFromLayersAnd3DToLayer(self):
        # Example: from_layer.shape = [3,2], to_layer.shape = [3,2,2] => weights.shape = [3,2,3]; axis_length = 1
        input_array = np.array([[1, 0], [1, 1], [0, 1]])
        weight_array = np.array([[[2, 1, 0], [2, 2, 3]],
                                 [[1, 1, 0], [1, 2, 0]],
                                 [[0, 1, 2], [1, 1, 1]]])
        transformation2 = Linear()
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([[[3, 1], [4, 5]],
                                    [[2, 1], [3, 2]],
                                    [[1, 3], [2, 2]]])
        self.assertTrue((actual_output == expected_output).all())

    def testThat_LinearTransformation_givesCorrectTransformation_with3DFromLayersAnd1DToLayer(self):
        # Example: from_layer.shape = [3,2,2], to_layer.shape = [4,] => weights.shape = [4,3,2,2]; axis_length = 3
        input_array = np.array([[[1, 0], [1, 1]],
                                [[0, 1], [1, 2]],
                                [[2, 1], [1, 1]]])
        weight_array = np.array([[[[2, 1], [2, 2]], [[1, 1], [1, 2]], [[0, 1], [1, 1]]],
                                 [[[2, 0], [2, 3]], [[1, 0], [1, 0]], [[0, 2], [1, 0]]],
                                 [[[1, 0], [2, 3]], [[1, 0], [2, 0]], [[1, 2], [1, 1]]],
                                 [[[2, 0], [2, 3]], [[1, 1], [2, 0]], [[0, 2], [1, 1]]]])
        transformation2 = Linear()
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([15, 11, 14, 14])
        self.assertTrue((actual_output == expected_output).all())

    def testThat_LinearTransformation_givesCorrectBackPropagationDelta_with1DLayers(self):
        # Example: from_layer.shape = [3,], to_layer.shape = [2,] => weights.shape = [2,3]; axis_length = 1
        transformed_input_array = np.array([1, 0, 1])
        activation = Sigmoid()
        weight_array = np.array([[2, 1, 0], [2, 2, 3]])
        delta_vectors = np.array([0.1, 0.2])
        transformation3 = Linear()
        actual_output = transformation3.backpropagate_delta(delta_vectors, weight_array, activation,
                                                            transformed_input_array)
        expected_output = np.array([0.12, 0.12, 0.12])
        self.assertTrue((np.around(actual_output, 2) == expected_output).all())

    def testThat_LinearTransformation_givesCorrectBackPropagationDelta_with2DFromLayersAnd3DToLayer(self):
        # Example: from_layer.shape = [3,2], to_layer.shape = [2,3,2] => weights.shape = [2,]; axis_length = 0
        transformed_input_array = np.array([[1, 0], [0, 1], [1, 1]])
        activation = Sigmoid()
        weight_array = np.array([2, 3])
        delta_to_layer = np.array([[[0.1, 0.1], [0.2, 0.2], [0.2, 0.1]],
                                   [[0.2, 0.1], [0.3, 0.1], [0.1, 0.3]]])
        transformation3 = Linear()
        actual_output = transformation3.backpropagate_delta(delta_to_layer, weight_array, activation,
                                                            transformed_input_array)
        # tensordot = [[0.8, 0.5], [1.3, 0.7], [0.7, 1.1]]
        # activation_derivative = [[0.197, 0.25], [0.25, 0.197], [0.197, 0.197]]
        expected_output = np.array([[0.157, 0.125], [0.325, 0.138], [0.138, 0.216]])
        self.assertTrue((np.around(actual_output, 3) == expected_output).all())

    def testThat_LinearTransformation_givesCorrectBackPropagationDelta_with3DFromLayersAnd1DToLayer(self):
        # Example: from_layer.shape = [3,2,2], to_layer.shape = [4,] => weights.shape = [4,3,2,2]; axis_length = 1
        transformed_input_array = np.array([[[1, 0], [1, 1]],
                                            [[0, 1], [1, 2]],
                                            [[2, 1], [1, 1]]])
        weight_array = np.array([[[[2, 1], [2, 2]], [[1, 1], [1, 2]], [[0, 1], [1, 1]]],
                                 [[[2, 0], [2, 3]], [[1, 0], [1, 0]], [[0, 2], [1, 0]]],
                                 [[[1, 0], [2, 3]], [[1, 0], [2, 0]], [[1, 2], [1, 1]]],
                                 [[[2, 0], [2, 3]], [[1, 1], [2, 0]], [[0, 2], [1, 1]]]])
        activation = Sigmoid()
        delta_vectors = np.array([0.1, 0.2, 0.3, 0.1])
        transformation3 = Linear()
        actual_output = transformation3.backpropagate_delta(delta_vectors, weight_array, activation,
                                                            transformed_input_array)
        # tensordot = [[[1.1, 0.1], [1.4, 2.0]], [[0.7, 0.2], [1.1, 0.2]], [[0.3, 1.3], [0.7, 0.5]]]
        # activation_derivative = [[[0.197, 0.25], [0.197, 0.197]], [[0.25, 0.197], [0.197, 0.105]], [[0.105, 0.197], [0.197, 0.197]]]
        expected_output = np.array([[[0.216, 0.025], [0.275, 0.393]],
                                    [[0.175, 0.039], [0.216, 0.021]],
                                    [[0.031, 0.256], [0.138, 0.098]]])
        self.assertTrue((np.around(actual_output, 3) == expected_output).all())

    def testThat_LinearTransformation_givesBackPropagationDelta_inCorrectShape(self):
        activation = Sigmoid()
        transformation4 = Linear()
        # Example1: from_layer.shape = [3,2,2], to_layer.shape = [4,] => weights.shape = [4,3,2,2]; axis_length = 1
        transformed_input_array1 = np.array(np.arange(12)).reshape([3, 2, 2])
        weight_array1 = np.array(np.arange(48)).reshape([4, 3, 2, 2])
        delta_vectors1 = np.array(np.arange(4)).reshape([4])
        actual_output1 = transformation4.backpropagate_delta(delta_vectors1, weight_array1, activation,
                                                             transformed_input_array1)
        # Example2: from_layer.shape = [3,2,3,2,2], to_layer.shape = [4,2,2] => weights.shape = [4,3,2,3]; axis_length = 3
        transformed_input_array2 = np.array(np.arange(72)).reshape([3, 2, 3, 2, 2])
        weight_array2 = np.array(np.arange(72)).reshape([4, 3, 2, 3])
        delta_vectors2 = np.array(np.arange(16)).reshape([4, 2, 2])
        actual_output2 = transformation4.backpropagate_delta(delta_vectors2, weight_array2, activation,
                                                             transformed_input_array2)
        self.assertTrue(actual_output1.shape == (3, 2, 2))
        self.assertTrue(actual_output2.shape == (3, 2, 3, 2, 2))
