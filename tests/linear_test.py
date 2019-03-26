from source.linear import Linear
from source.activation import Sigmoid
import numpy as np
import unittest


class LinearTransformationClassTests(unittest.TestCase):
    @staticmethod
    def get_test_sample_data():
        input_shapes = [[10], [2], [2], [2], [2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3],
                        [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4],
                        [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [3, 2, 3, 2, 2]]
        output_shapes = [[10], [3], [3, 2], [3, 2, 2], [2, 3], [2], [3], [4, 3], [4, 2], [2, 3], [2, 2, 3], [2, 3, 3],
                         [3, 2, 2], [4], [2], [3, 4], [2, 4], [2, 3], [2, 3, 4], [3, 3, 4], [3, 4, 4], [3, 4, 5],
                         [2, 2, 3, 4], [2, 3, 3, 4], [2, 4, 4, 4], [2, 2, 2, 3], [4, 2, 2]]
        weights_shapes = [[1], [3, 2], [3], [3, 2], [2, 3, 2], [2, 2, 3], [2], [4, 2], [4, 2, 2, 3], [1], [2],
                          [2, 3, 2], [3, 2, 2, 2, 3], [2, 3], [2, 2, 3, 4], [2], [2, 2, 3], [2, 3, 2, 3, 4],
                          [1], [3, 2], [3, 4, 2, 3], [3, 4, 5, 2, 3, 4], [2], [2, 3, 2], [2, 4, 4, 2, 3],
                          [2, 2, 2, 3, 2, 3, 4], [4, 3, 2, 3]]
        forward_propagation_axes = [0, 1, 0, 0, 1, 2, 1, 1, 2, 0, 0, 1, 2, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3]
        transposed_weights_axes = [[0], [1, 0], [0], [0, 1], [2, 0, 1], [1, 2, 0], [0], [1, 0], [2, 3, 0, 1], [0],
                                   [0], [2, 0, 1], [3, 4, 0, 1, 2], [0, 1], [1, 2, 3, 0], [0], [1, 2, 0],
                                   [2, 3, 4, 0, 1], [0], [1, 0], [2, 3, 0, 1], [3, 4, 5, 0, 1, 2], [0],
                                   [2, 0, 1], [3, 4, 0, 1, 2], [4, 5, 6, 0, 1, 2, 3], [1, 2, 3, 0]]
        back_propagation_axes = [0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 1]
        transposed_input_axes = [[0], [0], [0], [0], [0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0],
                                 [0, 1], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [0, 1, 2], [1, 2, 0],
                                 [2, 0, 1], [0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [3, 4, 0, 1, 2]]
        gradient_axes = [1, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 2]
        return input_shapes, output_shapes, weights_shapes, forward_propagation_axes, transposed_weights_axes, back_propagation_axes, transposed_input_axes, gradient_axes

    # ========== FORWARD PROPAGATION PARAMETERS TESTS ========== #

    def testThat_getForwardPropagationParameters_givesCorrectDetails_forVariousInputAndOutputShapes(self):
        input_shapes, output_shapes, weights_shapes, forward_propagation_axes, _, _, _, _ = self.get_test_sample_data()
        actual_weight_shapes = list()
        actual_forward_propagation_axes = list()
        for i in range(len(input_shapes)):
            transformation1 = Linear("optimal", input_shapes[i], output_shapes[i])
            actual_weight_shapes.append(transformation1.weights_shape)
            actual_forward_propagation_axes.append(transformation1.forward_propagation_axes)
        self.assertEqual(forward_propagation_axes, actual_forward_propagation_axes)
        self.assertEqual(weights_shapes, actual_weight_shapes)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withFullyConnected1DLayers(self):
        transformation1 = Linear(connection_type="fully", input_shape=[10, ], output_shape=[10, ])
        expected_weights_shape = [10, 10]
        expected_forward_propagation_axes = 1
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withOptimallyConnected1DLayers(self):
        transformation1 = Linear(connection_type="optimal", input_shape=[10, ], output_shape=[10, ])
        expected_weights_shape = [1]
        expected_forward_propagation_axes = 0
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withFullyConnected1DInputAnd2DOutputLayers(self):
        transformation1 = Linear(connection_type="fully", input_shape=[10, ], output_shape=[5, 10])
        expected_weights_shape = [5, 10, 10]
        expected_forward_propagation_axes = 1
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withOptimallyConnected1DInputAnd2DOutputLayers(
            self):
        transformation1 = Linear(connection_type="optimal", input_shape=[10, ], output_shape=[5, 10])
        expected_weights_shape = [5]
        expected_forward_propagation_axes = 0
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withFullyConnected2DInputAnd3DOutputLayers(self):
        transformation1 = Linear(connection_type="fully", input_shape=[10, 5], output_shape=[5, 10, 5])
        expected_weights_shape = [5, 10, 5, 10, 5]
        expected_forward_propagation_axes = 2
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withOptimallyConnected2DInputAnd3DOutputLayers(
            self):
        transformation1 = Linear(connection_type="optimal", input_shape=[10, 5], output_shape=[5, 10, 5])
        expected_weights_shape = [5]
        expected_forward_propagation_axes = 0
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_with2DInputAnd3DOutputLayerButNoCommonality(self):
        # Here even optimal connection works like fully connected
        transformation1 = Linear(connection_type="optimal", input_shape=[10, 5], output_shape=[2, 3, 4])
        expected_weights_shape = [2, 3, 4, 10, 5]
        expected_forward_propagation_axes = 2
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    def testThat_getForwardPropagationParameters_givesCorrectDetails_withOptimallyConnected3DLayers(self):
        transformation1 = Linear(connection_type="optimal", input_shape=[10, 5, 4], output_shape=[2, 3, 4])
        expected_weights_shape = [2, 3, 10, 5]
        expected_forward_propagation_axes = 2
        self.assertEqual(expected_forward_propagation_axes, transformation1.forward_propagation_axes)
        self.assertEqual(expected_weights_shape, transformation1.weights_shape)

    # ========== FORWARD PROPAGATION VALUES TESTS ========== #

    def testThat_transformFunction_givesCorrectTransformation_withOptimallyConnected1DInputAnd2DOutputLayers(self):
        # Example: input_shape = [3,], output_shape = [3,2] => weights_shape = [3,2,3]; FP_axes = 1
        input_array = np.array([1, 0, 1])
        weight_array = np.array([[[2, 1, 0], [2, 2, 3]],
                                 [[1, 1, 0], [1, 2, 0]],
                                 [[0, 1, 2], [1, 1, 1]]])
        transformation2 = Linear(connection_type="optimal", input_shape=list(input_array.shape), output_shape=[3, 2])
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([[2, 5], [1, 1], [2, 2]])
        self.assertTrue((actual_output == expected_output).all())

    def testThat_transformFunction_givesCorrectTransformation_withOptimallyConnected2DInputAnd3DOutputLayers(self):
        # Example: input_shape = [3,2], output_shape= [3,2,2] => weights_shape = [3,2,3]; FP_axes = 1
        input_array = np.array([[1, 0], [1, 1], [0, 1]])
        weight_array = np.array([[[2, 1, 0], [2, 2, 3]],
                                 [[1, 1, 0], [1, 2, 0]],
                                 [[0, 1, 2], [1, 1, 1]]])
        transformation2 = Linear(connection_type="optimal", input_shape=list(input_array.shape), output_shape=[3, 2, 2])
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([[[3, 1], [4, 5]],
                                    [[2, 1], [3, 2]],
                                    [[1, 3], [2, 2]]])
        self.assertTrue((actual_output == expected_output).all())

    def testThat_transformFunction_givesCorrectTransformation_withOptimallyConnected3DInputAnd1DOutputLayers(self):
        # Example: input_shape = [3,2,2], output_shape = [4,] => weights_shape = [4,3,2,2]; FP_axes = 3
        input_array = np.array([[[1, 0], [1, 1]],
                                [[0, 1], [1, 2]],
                                [[2, 1], [1, 1]]])
        weight_array = np.array([[[[2, 1], [2, 2]], [[1, 1], [1, 2]], [[0, 1], [1, 1]]],
                                 [[[2, 0], [2, 3]], [[1, 0], [1, 0]], [[0, 2], [1, 0]]],
                                 [[[1, 0], [2, 3]], [[1, 0], [2, 0]], [[1, 2], [1, 1]]],
                                 [[[2, 0], [2, 3]], [[1, 1], [2, 0]], [[0, 2], [1, 1]]]])
        transformation2 = Linear(connection_type="optimal", input_shape=list(input_array.shape), output_shape=[4])
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([15, 11, 14, 14])
        self.assertTrue((actual_output == expected_output).all())

    def testThat_transformFunction_givesCorrectTransformation_withFullyConnected2DLayers(self):
        # Example: input_shape = [2,2], output_shape = [3,2] => weights_shape = [3,2,2,2]; FP_axes = 2
        input_array = np.array([[1, 0], [1, 1]])
        weight_array = np.array([[[[2, 1], [2, 2]], [[1, 1], [1, 2]]],
                                 [[[2, 0], [2, 3]], [[1, 0], [1, 0]]],
                                 [[[2, 0], [2, 3]], [[1, 1], [2, 0]]]])
        transformation2 = Linear(connection_type="fully", input_shape=list(input_array.shape), output_shape=[3, 2])
        actual_output = transformation2.transform(input_array, weight_array)
        expected_output = np.array([[6, 4], [7, 2], [7, 3]])
        self.assertTrue((np.around(actual_output, 3) == expected_output).all())

    # ========== BACK PROPAGATION PARAMETERS TESTS ========== #

    def testThat_getBackPropagationParameters_givesCorrectDetails_forVariousInputAndOutputShapes(self):
        input_shapes, output_shapes, _, _, transposed_weights_axes, back_propagation_axes, _, _ = self.get_test_sample_data()
        actual_weight_transposition_axes = list()
        actual_back_propagation_axes = list()
        for i in range(len(input_shapes)):
            transformation3 = Linear("optimal", input_shapes[i], output_shapes[i])
            actual_weight_transposition_axes.append(transformation3.transposed_weights_axes)
            actual_back_propagation_axes.append(transformation3.back_propagation_axes)
        self.assertEqual(transposed_weights_axes, actual_weight_transposition_axes)
        self.assertEqual(back_propagation_axes, actual_back_propagation_axes)

    # ========== BACK PROPAGATION VALUES TESTS ========== #

    def testThat_backPropagateDelta_givesCorrectDeltaDimensions_forVariousInputAndOutputShapes(self):
        input_shapes, output_shapes, weights_shapes, _, _, _, _, _ = self.get_test_sample_data()
        actual_delta_shapes = list()
        for i in range(len(input_shapes)):
            transformation4 = Linear("optimal", input_shapes[i], output_shapes[i])
            output_layer_delta = np.zeros(output_shapes[i])
            weights = np.zeros(weights_shapes[i])
            activation = Sigmoid()
            transformed_input = np.zeros(input_shapes[i])
            input_layer_delta = transformation4.back_propagate_delta(output_layer_delta, weights, activation,
                                                                     transformed_input)
            actual_delta_shapes.append(list(input_layer_delta.shape))

        self.assertEqual(input_shapes, actual_delta_shapes)

    def testThat_backPropagateDelta_givesCorrectDelta_withOptimallyConnected1DLayers(self):
        # Example: from_layer.shape = [3,], to_layer.shape = [2,] => weights.shape = [2,3]; axis_length = 1
        transformed_input = np.array([1, 0, 1])
        activation = Sigmoid()
        weights = np.array([[2, 1, 0], [2, 2, 3]])
        output_layer_delta = np.array([0.1, 0.2])
        expected_input_layer_delta = np.array([0.12, 0.12, 0.12])

        transformation4 = Linear("optimal", list(transformed_input.shape), list(output_layer_delta.shape))
        actual_input_layer_delta = transformation4.back_propagate_delta(output_layer_delta, weights, activation,
                                                                        transformed_input)
        self.assertTrue((np.around(actual_input_layer_delta, 2) == expected_input_layer_delta).all())

    def testThat_backPropagateDelta_givesCorrectDelta_withOptimallyConnected2DInputAnd3DOutputLayers(self):
        # Example: from_layer.shape = [3,2], to_layer.shape = [2,3,2] => weights.shape = [2,]; axis_length = 0
        transformed_input = np.array([[1, 0], [0, 1], [1, 1]])
        activation = Sigmoid()
        weights = np.array([2, 3])
        output_layer_delta = np.array([[[0.1, 0.1], [0.2, 0.2], [0.2, 0.1]],
                                       [[0.2, 0.1], [0.3, 0.1], [0.1, 0.3]]])
        # tensordot = [[0.8, 0.5], [1.3, 0.7], [0.7, 1.1]]
        # activation_derivative = [[0.197, 0.25], [0.25, 0.197], [0.197, 0.197]]
        expected_input_layer_delta = np.array([[0.157, 0.125], [0.325, 0.138], [0.138, 0.216]])

        transformation4 = Linear("optimal", list(transformed_input.shape), list(output_layer_delta.shape))
        actual_input_layer_delta = transformation4.back_propagate_delta(output_layer_delta, weights, activation,
                                                                        transformed_input)
        self.assertTrue((np.around(actual_input_layer_delta, 3) == expected_input_layer_delta).all())

    def testThat_backPropagateDelta_givesCorrectDelta_withOptimallyConnected3DInputAnd1DOutputLayers(self):
        # Example: from_layer.shape = [3,2,2], to_layer.shape = [4,] => weights.shape = [4,3,2,2]; axis_length = 1
        transformed_input = np.array([[[1, 0], [1, 1]],
                                      [[0, 1], [1, 2]],
                                      [[2, 1], [1, 1]]])
        activation = Sigmoid()
        weights = np.array([[[[2, 1], [2, 2]], [[1, 1], [1, 2]], [[0, 1], [1, 1]]],
                            [[[2, 0], [2, 3]], [[1, 0], [1, 0]], [[0, 2], [1, 0]]],
                            [[[1, 0], [2, 3]], [[1, 0], [2, 0]], [[1, 2], [1, 1]]],
                            [[[2, 0], [2, 3]], [[1, 1], [2, 0]], [[0, 2], [1, 1]]]])
        output_layer_delta = np.array([0.1, 0.2, 0.3, 0.1])
        # tensordot = [[[1.1, 0.1], [1.4, 2.0]], [[0.7, 0.2], [1.1, 0.2]], [[0.3, 1.3], [0.7, 0.5]]]
        # activation_derivative = [[[0.197, 0.25], [0.197, 0.197]], [[0.25, 0.197], [0.197, 0.105]], [[0.105, 0.197], [0.197, 0.197]]]
        expected_input_layer_delta = np.array([[[0.216, 0.025], [0.275, 0.393]],
                                               [[0.175, 0.039], [0.216, 0.021]],
                                               [[0.031, 0.256], [0.138, 0.098]]])
        transformation4 = Linear("optimal", list(transformed_input.shape), list(output_layer_delta.shape))
        actual_input_layer_delta = transformation4.back_propagate_delta(output_layer_delta, weights, activation,
                                                                        transformed_input)
        self.assertTrue((np.around(actual_input_layer_delta, 3) == expected_input_layer_delta).all())

    def testThat_backPropagateDelta_givesCorrectDelta_withFullyConnected2DLayers(self):
        # Example: from_layer.shape = [2,2], to_layer.shape = [3,2] => weights.shape = [3,2,2,2]; axis_length = 2
        transformed_input = np.array([[1, 0], [1, 1]])
        activation = Sigmoid()
        weights = np.array([[[[2, 1], [2, 2]], [[1, 1], [1, 2]]],
                            [[[2, 0], [2, 3]], [[1, 0], [1, 0]]],
                            [[[2, 0], [2, 3]], [[1, 1], [2, 0]]]])
        output_layer_delta = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.3]])
        # tensordot = [[1.8, 0.6], [2.1, 2.1]]
        # activation_derivative = [[0.197, 0.25], [0.197, 0.197]]
        expected_input_layer_delta = np.array([[0.354, 0.150], [0.413, 0.413]])
        transformation4 = Linear("fully", list(transformed_input.shape), list(output_layer_delta.shape))
        actual_input_layer_delta = transformation4.back_propagate_delta(output_layer_delta, weights, activation,
                                                                        transformed_input)
        self.assertTrue((np.around(actual_input_layer_delta, 3) == expected_input_layer_delta).all())

    # ========== GRADIENT PARAMETERS TESTS ========== #

    def testThat_getGradientParameters_givesDetails_forVariousInputAndOutputShapes(self):
        input_shapes, output_shapes, _, _, _, _, transposed_input_axes, gradient_axes = self.get_test_sample_data()
        actual_transposed_input_axes = list()
        actual_gradient_axes = list()
        for i in range(len(input_shapes)):
            transformation5 = Linear("optimal", input_shapes[i], output_shapes[i])
            actual_transposed_input_axes.append(transformation5.transposed_input_axes)
            actual_gradient_axes.append(transformation5.weight_gradient_axes)

        self.assertEqual(transposed_input_axes, actual_transposed_input_axes)
        self.assertEqual(gradient_axes, actual_gradient_axes)

    # ========== GRADIENT VALUES TESTS ========== #

    def testThat_LinearTransformation_givesCorrectWeightsGradient_with1DFullyConnectedLayers(self):
        # Example: from_layer.shape = [2,], to_layer.shape = [3,] => weights.shape = [3,2]; axis_length = 1
        output_layer_delta = np.array([1, 2, 3])
        activated_input = np.array([1, 2])
        old_method_gradients = np.dot(output_layer_delta.reshape([1, 3]).transpose(), activated_input.reshape([1, 2]))
        transformation6 = Linear("optimal", list(activated_input.shape), list(output_layer_delta.shape))
        actual_gradient_for_weights = transformation6.get_gradient_for_weights(output_layer_delta, activated_input)
        self.assertTrue((actual_gradient_for_weights == old_method_gradients).all())

    def testThat_LinearTransformation_givesCorrectWeightsGradient_with1DFromLayerAnd3DToLayer_optimalConnection(self):
        # Example: from_layer.shape = [2,], to_layer.shape = [3,2,2] => weights.shape = [3,2]; axis_length = 1
        output_layer_delta = np.array(np.arange(12)).reshape([3, 2, 2])
        activated_input = np.array([1, 2])
        expected_gradient_for_weights = np.array([[2, 8], [14, 20], [26, 32]])
        transformation6 = Linear("optimal", list(activated_input.shape), list(output_layer_delta.shape))
        actual_gradient_for_weights = transformation6.get_gradient_for_weights(output_layer_delta, activated_input)
        self.assertTrue((actual_gradient_for_weights == expected_gradient_for_weights).all())
