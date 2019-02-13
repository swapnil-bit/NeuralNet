from source.layer import Layer
# from source.transformation import Transformation, Linear
from source.connection import Connection
import numpy as np
import unittest


class ConnectionClassTests(unittest.TestCase):
    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_for1DLayers(self):
        from_layer = Layer(0, [3])
        to_layer = Layer(1, [4])
        connection = Connection(from_layer, to_layer)
        actual_weights_shape = connection.get_optimum_weights_shape()
        expected_weight_shape = [4, 3]
        self.assertEqual(expected_weight_shape, actual_weights_shape)

    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_for2DLayers(self):
        from_layer = Layer(0, [3, 4])
        to_layer = Layer(1, [4, 4])
        connection = Connection(from_layer, to_layer)
        actual_weights_shape = connection.get_optimum_weights_shape()
        expected_weight_shape = [4, 3]
        self.assertEqual(expected_weight_shape, actual_weights_shape)

    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_for1DInputLayersAnd2DOutputLayer(self):
        from_layer = Layer(0, [3])
        to_layer = Layer(1, [4, 3])
        connection = Connection(from_layer, to_layer)
        actual_weights_shape = connection.get_optimum_weights_shape()
        expected_weight_shape = [4, ]
        self.assertEqual(expected_weight_shape, actual_weights_shape)

    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_for2DInputLayersAnd1DOutputLayer(self):
        from_layer = Layer(0, [4, 3])
        to_layer = Layer(1, [3])
        connection = Connection(from_layer, to_layer)
        actual_weights_shape = connection.get_optimum_weights_shape()
        expected_weight_shape = [4, ]
        self.assertEqual(expected_weight_shape, actual_weights_shape)

    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_for2DInputLayersAnd3DOutputLayer(self):
        from_layer = Layer(0, [4, 3])
        to_layer = Layer(1, [2, 3, 3])
        connection = Connection(from_layer, to_layer)
        actual_weights_shape = connection.get_optimum_weights_shape()
        expected_weight_shape = [2, 3, 4]
        self.assertEqual(expected_weight_shape, actual_weights_shape)

    def testThat_getOptimumWeightsShape_calculatesShapeCorrectly_for3DInputLayersAnd2DOutputLayer(self):
        from_layer = Layer(0, [4, 3, 2])
        to_layer = Layer(1, [2, 4])
        connection = Connection(from_layer, to_layer)
        actual_weights_shape = connection.get_optimum_weights_shape()
        expected_weight_shape = [2, 4, 4, 3, 2]
        self.assertEqual(expected_weight_shape, actual_weights_shape)

    def testThat_getWeights_assignsWeightsCorrectly_for2DInputLayersAnd3DOutputLayer(self):
        from_layer = Layer(0, [4, 3])
        to_layer = Layer(1, [2, 3, 3])
        connection = Connection(from_layer, to_layer)
        actual_weights = connection.get_weights(np.array([]))
        expected_weights = np.zeros([2, 3, 3, 4, 3])
        self.assertTrue((expected_weights == actual_weights).all())

    def testThat_getWeights_assignsWeightsCorrectly_for3DInputLayersAnd2DOutputLayer(self):
        from_layer = Layer(0, [4, 3, 2])
        to_layer = Layer(1, [4, 2])
        connection = Connection(from_layer, to_layer, connection_type="Opt")
        actual_weights = connection.get_weights(np.ones([4, 4, 3]))
        expected_weights = np.ones([4, 4, 3])
        self.assertTrue((expected_weights == actual_weights).all())
