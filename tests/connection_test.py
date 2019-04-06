from source.layer import Layer
from source.connection import Connection
import numpy as np
import unittest


class ConnectionClassTests(unittest.TestCase):
    @staticmethod
    def assertArrayEqual(expected, actual):
        return np.testing.assert_array_equal(actual, expected)

    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_forVariousInputAndOutputShapes(self):
        input_shapes = [[10], [2], [2], [2], [2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3],
                        [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4],
                        [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [3, 2, 3, 2, 2]]
        output_shapes = [[10], [3], [3, 2], [3, 2, 2], [2, 3], [2], [3], [4, 3], [4, 2], [2, 3], [2, 2, 3], [2, 3, 3],
                         [3, 2, 2], [4], [2], [3, 4], [2, 4], [2, 3], [2, 3, 4], [3, 3, 4], [3, 4, 4], [3, 4, 5],
                         [2, 2, 3, 4], [2, 3, 3, 4], [2, 4, 4, 4], [2, 2, 2, 3], [4, 2, 2]]
        expected_weight_shapes = [[1], [3, 2], [3], [3, 2], [2, 3, 2], [2, 2, 3], [2], [4, 2], [4, 2, 2, 3], [1], [2],
                                  [2, 3, 2], [3, 2, 2, 2, 3], [2, 3], [2, 2, 3, 4], [2], [2, 2, 3], [2, 3, 2, 3, 4],
                                  [1], [3, 2], [3, 4, 2, 3], [3, 4, 5, 2, 3, 4], [2], [2, 3, 2], [2, 4, 4, 2, 3],
                                  [2, 2, 2, 3, 2, 3, 4], [4, 3, 2, 3]]
        actual_weights_shape = list()
        for i in range(len(input_shapes)):
            from_layer = Layer(0, input_shapes[i])
            to_layer = Layer(1, output_shapes[i])
            connection = Connection(from_layer, to_layer, connection_type="optimal")
            weights_shape = list(connection.weights.shape)
            actual_weights_shape.append(weights_shape)

        self.assertEqual(expected_weight_shapes, actual_weights_shape)

    def testThat_getWeights_assignsWeightsCorrectly_for2DInputAnd3DOutputLayers(self):
        from_layer = Layer(0, [4, 3])
        to_layer = Layer(1, [2, 3, 3])
        connection = Connection(from_layer, to_layer)
        actual_weights = connection.initialize_weights(np.array([]), "zeros")
        expected_weights = np.zeros([2, 3, 3, 4, 3])
        self.assertArrayEqual(expected_weights, actual_weights)

    def testThat_getWeights_assignsWeightsCorrectly_for3DInputAnd2DOutputLayers(self):
        from_layer = Layer(0, [4, 3, 2])
        to_layer = Layer(1, [4, 2])
        connection = Connection(from_layer, to_layer, connection_type="optimal")
        actual_weights = connection.initialize_weights(np.ones([4, 4, 3]), "zeros")
        expected_weights = np.ones([4, 4, 3])
        self.assertArrayEqual(expected_weights, actual_weights)
