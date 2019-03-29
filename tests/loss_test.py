from source.loss import CrossEntropy, MeanSquare
import numpy as np
import unittest


class LossClassTests(unittest.TestCase):
    def testThat_CrossEntropy_Function_givesCorrectOutput(self):
        actual_y_array = np.array([1, 0, 1])
        predicted_y_array1 = np.array([0.9, 0.1, 0.9])
        predicted_y_array2 = np.array([0.1, 0.9, 0.1])
        loss1 = CrossEntropy()
        cost1 = loss1.function(predicted_y_array1, actual_y_array)
        cost2 = loss1.function(predicted_y_array2, actual_y_array)
        self.assertTrue(cost1 < cost2)

    def testThat_CrossEntropy_DeltaFunction_givesCorrectOutput(self):
        actual_y_array = [np.array([1, 0, 1])]
        predicted_y_array = [np.array([0.9, 0.1, 0.9])]
        # expected_delta = [np.array([-0.1, 0.1, -0.1])]  # This doesn't work due to floating point accuracy error
        expected_delta = [predicted_y_array[i] - actual_y_array[i] for i in range(len(predicted_y_array))]
        loss1 = CrossEntropy()
        actual_delta = loss1.get_delta_last_layer(predicted_y_array, actual_y_array)
        self.assertTrue((actual_delta[0] == expected_delta[0]).all())

    def testThat_MeanSquare_Function_givesCorrectOutput(self):
        actual_y_array = np.array([1, 0, 1])
        predicted_y_array = np.array([0.9, 0.1, 0.9])
        loss2 = MeanSquare()
        actual_cost = loss2.function(predicted_y_array, actual_y_array)
        expected_cost = 0.01
        self.assertEqual(np.around(actual_cost, 2), expected_cost)  # Not rounding gives floating point errors

    def testThat_MeanSquare_DeltaFunction_givesCorrectOutput(self):
        actual_y_array = [np.array([1, 0, 1])]
        predicted_y_array = [np.array([0.9, 0.1, 0.9])]
        expected_delta = [predicted_y_array[i] - actual_y_array[i] for i in range(len(predicted_y_array))]
        loss2 = MeanSquare()
        actual_delta = loss2.get_delta_last_layer(predicted_y_array, actual_y_array)
        self.assertTrue((actual_delta[0] == expected_delta[0]).all())
