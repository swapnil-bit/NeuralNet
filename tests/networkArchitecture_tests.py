from codeBase.networkArchitecture import NetworkArchitecture
from codeBase.layer import Layer
import numpy as np
import unittest


class NetworkClassTests(unittest.TestCase):
    def sigmoid_activation(self, input: np.array) -> np.array:
        return 1.0/(1.0 + np.exp(-input))

    def sigmoid_derivative(self, input: np.array) -> np.array:
        return self.sigmoid_activation(input)*(1-self.sigmoid_activation(input))

    def create_list_of_layers_with_sigmoid_activations(self, layer_sizes: [int]) -> [Layer]:
        layers_list = list()
        for size in layer_sizes:
            new_layer = Layer(size, self.sigmoid_activation, self.sigmoid_derivative)
            layers_list.append(new_layer)
        return layers_list

    def testThat_createAllLayers_createsLayersWithCorrect_predecessorsAndSuccessors(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 3)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_predecessors_of_all_layers = [[], [0], [0], [1,2]]
        expected_successors_of_all_layers = [[1,2], [3], [3], []]
        actual_predecessors_of_all_layers = [network1.all_layers[index].predecessors for index in range(4)]
        actual_successors_of_all_layers = [network1.all_layers[index].successors for index in range(4)]

        self.assertEqual(expected_predecessors_of_all_layers, actual_predecessors_of_all_layers)
        self.assertEqual(expected_successors_of_all_layers, actual_successors_of_all_layers)

    def testThat_feedForwardSequence_givesCorrectList(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 3, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_sequence = [0, 1, 2, 3, 4]
        actual_sequence = network1.feed_forward_sequence
        self.assertEqual(expected_sequence, actual_sequence)

    def testThat_backPropagationSequence_givesCorrectList(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 3, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_sequence = [4, 3, 2, 1, 0]
        actual_sequence = network1.back_propagation_sequence
        self.assertEqual(expected_sequence, actual_sequence)

    def testThat_getInputWeights_givesWeightsInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_weights = [np.zeros([3,2]), np.zeros([3,3])]
        actual_weights = network1.get_input_weights_of_a_layer(4)
        self.assertTrue((expected_weights[0] == actual_weights[0]).all())
        self.assertTrue((expected_weights[1] == actual_weights[1]).all())

    def testThat_getInputVector_givesVectorsInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_vectors = [np.zeros([3,1]), np.zeros([3,1])]
        actual_vectors = network1.get_input_weights_of_a_layer(4)
        self.assertTrue((expected_vectors[0] == actual_vectors[0]).all())
        self.assertTrue((expected_vectors[1] == actual_vectors[1]).all())

    def testThat_feedForwardAllLayers_givesOutputInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 4])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_last_layer_output = np.array([0.5] * 4).reshape(4, 1)
        network1.feed_forward_all_layers(np.array([[0,0,0]]).transpose())
        self.assertTrue((expected_last_layer_output == network1.all_layers[4].activated_output).all())

    def testThat_getOutputWeights_givesWeightsInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_weights = [np.zeros([3,3]), np.zeros([2,3])]
        actual_weights = network1.get_output_weights_of_a_layer(0)
        self.assertTrue((expected_weights[0] == actual_weights[0]).all())
        self.assertTrue((expected_weights[1] == actual_weights[1]).all())

    def testThat_getSuccessorsDeltas_givesDeltasInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_deltas = [np.zeros([3,1]), np.zeros([2,1])]
        actual_deltas = network1.get_successor_deltas_of_a_layer(0)
        self.assertTrue((expected_deltas[0] == actual_deltas[0]).all())
        self.assertTrue((expected_deltas[1] == actual_deltas[1]).all())

    def testThat_updateDeltas_flowsCorrectly(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.update_deltas_of_all_layers(np.zeros([3,1]), np.ones([1,1]))
        expected_delta = np.zeros([3,1])
        self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    def testThat_backPropagate_flowsCorrectly(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        training_data = [(np.zeros([3,1]), np.ones([1,1]))]
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
        self.assertTrue((np.zeros([3,1]) == gradient_for_biases[0]).all())
        self.assertTrue((np.zeros([2,3]) == gradient_for_weights[1]).all())
