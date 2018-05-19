from codeBase.networkConfigurations import NetworkConfigurations
import numpy as np
import unittest


class NetworkClassTests(unittest.TestCase):
    def testThat_crossEntropyCostFunction_givesCorrectOutput(self):
        actual_y_array = np.array([1,0,1])
        predicted_y_array1 = np.array([0.9, 0.1, 0.9])
        predicted_y_array2 = np.array([0.1, 0.9, 0.1])
        config1 = NetworkConfigurations()
        cost1 = config1.cross_entropy_cost_function(predicted_y_array1, actual_y_array)
        cost2 = config1.cross_entropy_cost_function(predicted_y_array2, actual_y_array)
        self.assertTrue((cost1 < cost2).all())