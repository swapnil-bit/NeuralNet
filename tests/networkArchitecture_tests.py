# from sklearn.datasets import fetch_mldata

from source.networkArchitecture import NetworkArchitecture
from source.layer import Layer
from source.activation import Sigmoid
from source.transformation import Linear
import numpy as np
import unittest


class NetworkClassTests(unittest.TestCase):
    def create_list_of_layers_with_sigmoid_activations(self, layer_sizes: [int]) -> [Layer]:
        layers_list = list()
        for size in layer_sizes:
            new_layer = Layer(size, input_transformation=Linear(), activation=Sigmoid())
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

    # def test_mnist_test(self):
    #     all_data = fetch_mldata(data_home="/Users/Swapnil/Analytics/", dataname="mnist-original")
    #     random_index = np.random.choice(len(all_data.data), 50000, replace=False)
    #     random_index1 = random_index[:40000]
    #     random_index2 = random_index[40000:]
    #     training_data_x = all_data.data[random_index1]/25500
    #     training_data_x = training_data_x.reshape(len(training_data_x), 1, 784)
    #
    #     training_data_target = all_data.target[random_index1].astype(int)
    #     training_data_y = np.zeros([len(training_data_target), 1, 10], dtype = int)
    #     for i in range(len(training_data_target)):
    #         training_data_y[i, 0, training_data_target[i]] = 1
    #
    #     training_data = tuple(zip(training_data_x, training_data_y))
    #     learning_rate = 0.5
    #     epochs = 30
    #     batch_size = 25
    #     layers_list = self.create_list_of_layers_with_sigmoid_activations([784, 100, 10])
    #     mnistNw = NetworkArchitecture(layers_list, [(0,1), (1,2)])
    #     mnistNw.train_netwrok(training_data, epochs, batch_size, learning_rate)
    #
    #     ## Prediction on test data
    #     test_data_x = all_data.data[random_index2]/25500
    #     test_data_x = test_data_x.reshape(len(test_data_x), 1, 784)
    #     test_data_target = all_data.target[random_index2].astype(int)
    #     output_digit = []
    #     for i in range(10000):
    #         mnistNw.feed_forward_all_layers(test_data_x[i])
    #         output = mnistNw.all_layers[2].activated_output
    #         output_digit.append(np.argmax(output))
    #     accuracy = sum(int(x == y) for (x, y) in tuple(zip(output_digit, test_data_target)))
    #     accuracy = accuracy/10000
    #     print(accuracy)