from codeBase.networkArchitecture import NetworkArchitecture
import numpy as np
import pandas as pd
import unittest


class NetworkClassTests(unittest.TestCase):
    def testThat_createAllLayers_createsLayersWithCorrect_predecessorsAndSuccessors(self):
        network1 = NetworkArchitecture([3, 3, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])
        expected_predecessors_of_all_layers = [[], [0], [0], [1,2]]
        expected_successors_of_all_layers = [[1,2], [3], [3], []]
        actual_predecessors_of_all_layers = [network1.all_layers[index].predecessors for index in range(4)]
        actual_successors_of_all_layers = [network1.all_layers[index].successors for index in range(4)]

        self.assertEqual(expected_predecessors_of_all_layers, actual_predecessors_of_all_layers)
        self.assertEqual(expected_successors_of_all_layers, actual_successors_of_all_layers)

    def testThat_feedForwardSequence_givesCorrectList(self):
        network1 = NetworkArchitecture([3, 3, 3, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_sequence = [0, 1, 2, 3, 4]
        actual_sequence = network1.feed_forward_sequence
        self.assertEqual(expected_sequence, actual_sequence)

    def testThat_backPropagationSequence_givesCorrectList(self):
        network1 = NetworkArchitecture([3, 3, 3, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_sequence = [4, 3, 2, 1, 0]
        actual_sequence = network1.back_propagation_sequence
        self.assertEqual(expected_sequence, actual_sequence)

    def testThat_getInputWeights_givesWeightsInCorrectDimensions(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_weights = [np.zeros([3,2]), np.zeros([3,3])]
        actual_weights = network1.get_input_weights_of_a_layer(4)
        self.assertTrue((expected_weights[0] == actual_weights[0]).all())
        self.assertTrue((expected_weights[1] == actual_weights[1]).all())

    def testThat_getInputVector_givesVectorsInCorrectDimensions(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_vectors = [np.zeros([3,1]), np.zeros([3,1])]
        actual_vectors = network1.get_input_weights_of_a_layer(4)
        self.assertTrue((expected_vectors[0] == actual_vectors[0]).all())
        self.assertTrue((expected_vectors[1] == actual_vectors[1]).all())

    def testThat_feedForwardAllLayers_givesOutputInCorrectDimensions(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 4], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_last_layer_output = np.array([0.5] * 4).reshape(4, 1)
        network1.feed_forward_all_layers(np.array([[0,0,0]]).transpose())
        self.assertTrue((expected_last_layer_output == network1.all_layers[4].activated_output).all())

    def testThat_getOutputWeights_givesWeightsInCorrectDimensions(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_weights = [np.zeros([3,3]), np.zeros([2,3])]
        actual_weights = network1.get_output_weights_of_a_layer(0)
        self.assertTrue((expected_weights[0] == actual_weights[0]).all())
        self.assertTrue((expected_weights[1] == actual_weights[1]).all())

    def testThat_getSuccessorsDeltas_givesDeltasInCorrectDimensions(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        expected_deltas = [np.zeros([3,1]), np.zeros([2,1])]
        actual_deltas = network1.get_successor_deltas_of_a_layer(0)
        self.assertTrue((expected_deltas[0] == actual_deltas[0]).all())
        self.assertTrue((expected_deltas[1] == actual_deltas[1]).all())

    def testThat_updateDeltas_flowsCorrectly(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        network1.update_deltas_of_all_layers(np.zeros([3,1]), np.ones([1,1]))
        expected_delta = np.zeros([3,1])
        self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    def testThat_backPropagate_flowsCorrectly(self):
        network1 = NetworkArchitecture([3, 3, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        training_data = [(np.zeros([3,1]), np.ones([1,1]))]
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
        self.assertTrue((np.zeros([3,1]) == gradient_for_biases[0]).all())
        self.assertTrue((np.zeros([2,3]) == gradient_for_weights[1]).all())

    def testThat_integrationWorksProperly(self):
        training_data = pd.read_csv("/Users/Swapnil/Analytics/Titanic/train.csv")
        training_data = training_data[["Survived", "Pclass", "Sex", "Age"]]
        training_data.loc[training_data.Sex == "male", 'Sex'] = 0
        training_data.loc[training_data.Sex == "female", 'Sex'] = 1
        training_data = training_data.fillna(0)
        training_data = training_data[training_data.Age != 0]
        training_data = training_data.loc[200:]
        y = training_data.Survived
        x = np.array(training_data[["Pclass", "Sex", "Age"]].values)
        x = x.reshape(len(training_data), 3, 1)
        new_training_data = tuple(zip(x, y))
        learning_rate = 0.001
        ttncNw = NetworkArchitecture([3, 1], [(0,1)])
        ttncNw.networkTraining(new_training_data, 500, learning_rate)
        print(ttncNw.weights, ttncNw.all_layers[1].bias)


