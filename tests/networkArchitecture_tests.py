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
        expected_weights = np.zeros([3,5])
        actual_weights = network1.get_input_weights_of_a_layer(4)
        self.assertTrue((expected_weights[0] == actual_weights[0]).all())
        self.assertTrue((expected_weights[1] == actual_weights[1]).all())

    def testThat_getInputVector_givesVectorsInCorrectDimensions_ForSingleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.all_layers[2].activated_output = np.zeros([1, 3])
        network1.all_layers[3].activated_output = np.zeros([1, 2])
        expected_vectors = np.zeros([1,5])
        actual_vectors = network1.get_input_vectors_of_a_layer(4)
        self.assertTrue((expected_vectors == actual_vectors).all())

    def testThat_getInputVector_givesVectorsInCorrectDimensions_ForMultipleInputs(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 3)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.all_layers[1].activated_output = np.zeros([4, 3])
        network1.all_layers[2].activated_output = np.ones([4, 2])
        expected_vectors = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
        actual_vectors = network1.get_input_vectors_of_a_layer(3)
        self.assertTrue((expected_vectors == actual_vectors).all())

    def testThat_feedForwardAllLayers_givesOutputInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 4])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_last_layer_output = np.array([0.5] * 4).reshape(4, 1)
        network1.feed_forward_all_layers(np.array([[0,0,0]]))
        self.assertTrue((expected_last_layer_output == network1.all_layers[4].activated_output).all())

    def testThat_getOutputWeights_givesWeightsInCorrectDimensions(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        expected_weights = np.zeros([5,3])
        actual_weights = network1.get_output_weights_of_a_layer(0)
        self.assertTrue((expected_weights == actual_weights).all())

    def testThat_getSuccessorsDeltas_givesDeltasInCorrectDimensions_WithSingleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.all_layers[1].delta = np.zeros([1, 3])
        network1.all_layers[2].delta = np.zeros([1, 2])
        expected_deltas = np.zeros([1, 5])
        actual_deltas = network1.get_successor_deltas_of_a_layer(0)
        self.assertTrue((expected_deltas == actual_deltas).all())

    def testThat_getSuccessorsDeltas_givesDeltasInCorrectDimensions_WithMultipleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.all_layers[1].delta = np.zeros([4, 3])
        network1.all_layers[2].delta = np.zeros([4, 2])
        expected_deltas = np.zeros([4, 5])
        actual_deltas = network1.get_successor_deltas_of_a_layer(0)
        self.assertTrue((expected_deltas == actual_deltas).all())

    def testThat_updateDeltas_flowsCorrectly_withSingleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.update_deltas_of_all_layers(np.zeros([1,3]), np.ones([1,1]))
        expected_delta = np.zeros([1,3])
        self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    def testThat_updateDeltas_flowsCorrectly_withMultipleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        network1.update_deltas_of_all_layers(np.zeros([4,3]), np.ones([4,1]))
        expected_delta = np.zeros([1,3])
        self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    def testThat_backPropagate_flowsCorrectly_withSingleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        training_data = [(np.zeros([1,3]), np.ones([1,1]))]
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
        self.assertTrue((np.zeros([3,1]) == gradient_for_biases[0]).all())
        self.assertTrue((np.zeros([2,3]) == gradient_for_weights[1]).all())

    def testThat_backPropagate_flowsCorrectly_withMultipleInput(self):
        layers_list = self.create_list_of_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        network1 = NetworkArchitecture(layers_list, layer_connection)
        training_data = [(np.zeros([1,3]), np.ones([1,1])), (np.zeros([1,3]), np.ones([1,1])),
                         (np.zeros([1,3]), np.ones([1,1])), (np.zeros([1,3]), np.ones([1,1]))]
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
        self.assertTrue((np.zeros([3,1]) == gradient_for_biases[0]).all())
        self.assertTrue((np.zeros([2,3]) == gradient_for_weights[1]).all())
