from source.loss import CrossEntropy, MeanSquare
import numpy as np
import unittest


class LossClassTests(unittest.TestCase):
    @staticmethod
    def assertArrayEqual(expected, actual):
        return np.testing.assert_array_equal(actual, expected)

    def testThat_CrossEntropy_Function_givesCorrectOutput_for1DArrayAndSingleBatchSize(self):
        actual_y_array = [np.array([1, 0, 1])]
        predicted_y_array = [np.array([0.9, 0.1, 0.9])]
        expected_cost = 0.1054
        loss1 = CrossEntropy()
        actual_cost = loss1.function(predicted_y_array, actual_y_array)
        self.assertEqual(expected_cost, np.round(actual_cost[0], 4))

    def testThat_CrossEntropy_Function_givesCorrectOutput_for1DArrayAndMultipleBatchSize(self):
        batch_size = 2
        actual_y_array = [np.array([1, 0, 1])] * batch_size
        predicted_y_array = [np.array([0.9, 0.1, 0.9]), np.array([0.1, 0.9, 0.1])]
        expected_cost = [0.1054, 2.3026]
        loss1 = CrossEntropy()
        actual_cost = loss1.function(predicted_y_array, actual_y_array)
        for i in range(batch_size):
            self.assertEqual(expected_cost[i], np.round(actual_cost[i], 4))

    def testThat_CrossEntropy_Function_givesCorrectOutput_for2DArrayAndSingleBatchSize(self):
        actual_y_array = [np.array([[1, 0, 1], [1,0, 1]])]
        predicted_y_array = [np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])]
        expected_cost = 1.2040
        loss1 = CrossEntropy()
        actual_cost = loss1.function(predicted_y_array, actual_y_array)
        self.assertEqual(expected_cost, np.round(actual_cost[0], 4))

    def testThat_CrossEntropy_Function_givesCorrectOutput_for2DArrayAndMultipleBatchSize(self):
        batch_size = 2
        actual_y_array = [np.array([[1, 0, 1], [1,0, 1]])] * batch_size
        predicted_y_array = [np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])] * batch_size
        expected_cost = [1.2040, 1.2040]
        loss1 = CrossEntropy()
        actual_cost = loss1.function(predicted_y_array, actual_y_array)
        for i in range(batch_size):
            self.assertEqual(expected_cost[i], np.round(actual_cost[i], 4))

    def testThat_CrossEntropy_DeltaFunction_givesCorrectOutput(self):
        actual_y_array = [np.array([1, 0, 1])]
        predicted_y_array = [np.array([0.9, 0.1, 0.9])]
        # expected_delta = [np.array([-0.1, 0.1, -0.1])]  # This doesn't work due to floating point accuracy error
        expected_delta = [predicted_y_array[i] - actual_y_array[i] for i in range(len(predicted_y_array))]
        loss1 = CrossEntropy()
        actual_delta = loss1.get_delta_last_layer(predicted_y_array, actual_y_array)
        self.assertArrayEqual(expected_delta[0], actual_delta[0])

    def testThat_MeanSquare_Function_givesCorrectOutput(self):
        actual_y_array = [np.array([1, 0, 1])]
        predicted_y_array = [np.array([0.9, 0.1, 0.9])]
        expected_cost = 0.01
        loss2 = MeanSquare()
        actual_cost = loss2.function(predicted_y_array, actual_y_array)
        self.assertEqual(expected_cost, np.around(actual_cost[0], 2))  # Not rounding gives floating point errors

    def testThat_MeanSquare_DeltaFunction_givesCorrectOutput(self):
        actual_y_array = [np.array([1, 0, 1])]
        predicted_y_array = [np.array([0.9, 0.1, 0.9])]
        expected_delta = [predicted_y_array[i] - actual_y_array[i] for i in range(len(predicted_y_array))]
        loss2 = MeanSquare()
        actual_delta = loss2.get_delta_last_layer(predicted_y_array, actual_y_array)
        self.assertArrayEqual(expected_delta[0], actual_delta[0])
