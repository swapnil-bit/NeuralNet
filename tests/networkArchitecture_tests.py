# from sklearn.datasets import fetch_mldata

from source.networkArchitecture import NetworkArchitecture
from source.layer import Layer
from source.connection import Connection
from source.activation import Sigmoid
from source.linear import Linear
import numpy as np
import unittest


class NetworkClassTests(unittest.TestCase):
    @staticmethod
    def create_list_of_1D_layers_with_sigmoid_activations(layers_shapes: [int]) -> [Layer]:
        layers_list = list()
        for index in range(len(layers_shapes)):
            new_layer = Layer(id=index, shape=[layers_shapes[index]], activation=Sigmoid())
            layers_list.append(new_layer)
        return layers_list

    @staticmethod
    def create_list_of_1D_layers_with_2D_structures_and_sigmoid_activations(layers_shapes: [int]) -> [Layer]:
        layers_list = list()
        for index in range(len(layers_shapes)):
            new_layer = Layer(id=index, shape=[layers_shapes[index], 1], activation=Sigmoid())
            layers_list.append(new_layer)
        return layers_list

    @staticmethod
    def get_layer_from_layer_id(id, layers_list):
        for layer in layers_list:
            if layer.id == id:
                return layer
        return None

    def create_list_of_connections_having_fully_connection_type(self, layers_list: [Layer],
                                                                layer_connections: [tuple],
                                                                weight_distribution: str = "zeros") -> [Connection]:
        connections_list = list()
        for connection in layer_connections:
            input_layer = self.get_layer_from_layer_id(connection[0], layers_list)
            output_layer = self.get_layer_from_layer_id(connection[1], layers_list)
            transformation = Linear("fully", input_layer.shape, output_layer.shape)
            new_connection = Connection(input_layer, output_layer, "fully", transformation, np.array([]),
                                        weight_distribution)
            connections_list.append(new_connection)
        return connections_list

    def testThat_createAllLayers_createsLayersWithCorrect_predecessorsAndSuccessors(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 3)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        expected_predecessors_of_all_layers = [[], [0], [0], [1, 2]]
        expected_successors_of_all_layers = [[1, 2], [3], [3], []]
        actual_predecessors_of_all_layers = [network1.all_layers[index].predecessors for index in range(4)]
        actual_successors_of_all_layers = [network1.all_layers[index].successors for index in range(4)]

        self.assertEqual(expected_predecessors_of_all_layers, actual_predecessors_of_all_layers)
        self.assertEqual(expected_successors_of_all_layers, actual_successors_of_all_layers)

    def testThat_feedForwardSequence_givesCorrectList(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 3, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        expected_sequence = [0, 1, 2, 3, 4]
        actual_sequence = network1.feed_forward_sequence
        self.assertEqual(expected_sequence, actual_sequence)

    def testThat_backPropagationSequence_givesCorrectList(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 3, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        expected_sequence = [4, 3, 2, 1, 0]
        actual_sequence = network1.back_propagation_sequence
        self.assertEqual(expected_sequence, actual_sequence)

    def testThat_getInputArrays_givesArraysInCorrectDimensions_ForSingleInput(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        network1.all_layers[2].activated_output = np.zeros([1, 3])
        network1.all_layers[3].activated_output = np.zeros([1, 2])
        expected_arrays = [np.zeros([1, 3]), np.zeros([1, 3])]
        actual_arrays = network1.get_input_arrays_of_a_layer(4)
        self.assertTrue((expected_arrays[0] == actual_arrays[0]).all())
        self.assertTrue((expected_arrays[1] == actual_arrays[1]).all())

    # def testThat_getInputVector_givesVectorsInCorrectDimensions_ForMultipleInputs(self):
    #     layers_list = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3])
    #     layer_connection = [(0, 1), (0, 2), (1, 3), (2, 3)]
    #     network1 = NetworkArchitecture(layers_list, layer_connection)
    #     network1.all_layers[1].activated_output = np.zeros([4, 3])
    #     network1.all_layers[2].activated_output = np.ones([4, 2])
    #     expected_vectors = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
    #     actual_vectors = network1.get_input_vectors_of_a_layer(3)
    #     self.assertTrue((expected_vectors == actual_vectors).all())

    def testThat_feedForwardAllLayers_givesOutputInCorrectDimensions(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 4])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        expected_last_layer_output = np.array([0.5] * 4).reshape(4, 1)
        network1.feed_forward_all_layers(np.array([0, 0, 0]))
        self.assertTrue((expected_last_layer_output == network1.all_layers[4].output_array).all())

    # def testThat_getSuccessorsDeltas_givesDeltasInCorrectDimensions_WithMultipleInput(self):
    #     layers_list = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
    #     layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
    #     network1 = NetworkArchitecture(layers_list, layer_connection)
    #     network1.all_layers[1].delta = np.zeros([4, 3])
    #     network1.all_layers[2].delta = np.zeros([4, 2])
    #     expected_deltas = np.zeros([4, 5])
    #     actual_deltas = network1.get_successor_deltas_of_a_layer(0)
    #     self.assertTrue((expected_deltas == actual_deltas).all())
    #
    def testThat_updateDeltas_flowsCorrectly_withSingleInput(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        network1.update_deltas_of_all_layers(np.zeros([3]), np.ones([3]))
        expected_delta = np.zeros([1, 3])
        self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    def testThat_updateDeltas_flowsCorrectly_withSingleInput_whenLastLayerHasSingleNeuron(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 1])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        network1.update_deltas_of_all_layers(np.zeros([3]), np.ones([1]))
        expected_delta = np.zeros([1, 3])
        self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    # def testThat_updateDeltas_flowsCorrectly_withMultipleInput(self):
    #     layers_list = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
    #     layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
    #     network1 = NetworkArchitecture(layers_list, layer_connection)
    #     network1.update_deltas_of_all_layers(np.zeros([4, 3]), np.ones([4, 1]))
    #     expected_delta = np.zeros([1, 3])
    #     self.assertTrue((expected_delta == network1.all_layers[0].delta).all())

    def testThat_backPropagate_flowsCorrectly_withSingleInput(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        training_data = [(np.zeros([3]), np.ones([3]))]
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
        self.assertTrue((np.zeros([3, 1]) == gradient_for_biases[0]).all())
        self.assertTrue((np.zeros([2, 3]) == gradient_for_weights[(0, 2)]).all())

    def testThat_backPropagate_flowsCorrectly_withSingleInput_whenLastLayerHasSingleNeuron(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 1, 2, 3, 1])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        training_data = [(np.zeros([3]), np.ones([1]))]
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
        self.assertTrue((np.zeros([3, 1]) == gradient_for_biases[0]).all())
        self.assertTrue((np.zeros([2, 3]) == gradient_for_weights[(0, 2)]).all())

    # def testThat_backPropagate_flowsCorrectly_withMultipleInput(self):
    #     layers_list = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
    #     layer_connection = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
    #     network1 = NetworkArchitecture(layers_list, layer_connection)
    #     training_data = [(np.zeros([1, 3]), np.ones([1, 1])), (np.zeros([1, 3]), np.ones([1, 1])),
    #                      (np.zeros([1, 3]), np.ones([1, 1])), (np.zeros([1, 3]), np.ones([1, 1]))]
    #     gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data)
    #     self.assertTrue((np.zeros([3, 1]) == gradient_for_biases[0]).all())
    #     self.assertTrue((np.zeros([2, 3]) == gradient_for_weights[1]).all())

    # def test_mnist_test(self):
    #     all_data = fetch_mldata(data_home="/Users/Swapnil/Analytics/Personal Projects/MNIST Exp/",
    #                             dataname="mnist-original")
    #     random_index = np.random.choice(len(all_data.data), 50000, replace=False)
    #     random_index1 = random_index[:40000]
    #     random_index2 = random_index[40000:]
    #     training_data_x = all_data.data[random_index1] / 25500
    #     training_data_x = training_data_x.reshape(len(training_data_x), 784)
    #
    #     training_data_target = all_data.target[random_index1].astype(int)
    #     training_data_y = np.zeros([len(training_data_target), 10], dtype=int)
    #     for i in range(len(training_data_target)):
    #         training_data_y[i, training_data_target[i]] = 1
    #
    #     training_data = tuple(zip(training_data_x, training_data_y))
    #     learning_rate = 0.5
    #     epochs = 30
    #     batch_size = 1
    #     layers = self.create_list_of_1D_layers_with_sigmoid_activations([784, 100, 10])
    #     connection_indices = [(0, 1), (1, 2)]
    #     connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices, "normal")
    #     mnistNw = NetworkArchitecture(layers, connections)
    #     mnistNw.train_netwrok(training_data, epochs, batch_size, learning_rate)
    #
    #     ## Prediction on test data
    #     test_data_x = all_data.data[random_index2] / 25500
    #     test_data_x = test_data_x.reshape(len(test_data_x), 784)
    #     test_data_target = all_data.target[random_index2].astype(int)
    #     output_digit = []
    #     for i in range(10000):
    #         mnistNw.feed_forward_all_layers(test_data_x[i])
    #         output = mnistNw.all_layers[2].output_array
    #         output_digit.append(np.argmax(output))
    #     accuracy = sum(int(x == y) for (x, y) in tuple(zip(output_digit, test_data_target)))
    #     accuracy = accuracy / 10000
    #     print(accuracy)
    #
    # def test_mnist_test_2D_structure(self):
    #     all_data = fetch_mldata(data_home="/Users/Swapnil/Analytics/Personal Projects/MNIST Exp/",
    #                             dataname="mnist-original")
    #     random_index = np.random.choice(len(all_data.data), 50000, replace=False)
    #     random_index1 = random_index[:40000]
    #     random_index2 = random_index[40000:]
    #     training_data_x = all_data.data[random_index1] / 25500
    #     training_data_x = training_data_x.reshape(len(training_data_x), 784, 1)
    #
    #     training_data_target = all_data.target[random_index1].astype(int)
    #     training_data_y = np.zeros([len(training_data_target), 10, 1], dtype=int)
    #     for i in range(len(training_data_target)):
    #         training_data_y[i, training_data_target[i]] = 1
    #
    #     training_data = tuple(zip(training_data_x, training_data_y))
    #     learning_rate = 0.5
    #     epochs = 30
    #     batch_size = 1
    #     layers = self.create_list_of_1D_layers_with_2D_structures_and_sigmoid_activations([784, 100, 10])
    #     connection_indices = [(0, 1), (1, 2)]
    #     connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices, "normal")
    #     mnistNw = NetworkArchitecture(layers, connections)
    #     mnistNw.train_netwrok(training_data, epochs, batch_size, learning_rate)
    #
    #     ## Prediction on test data
    #     test_data_x = all_data.data[random_index2] / 25500
    #     test_data_x = test_data_x.reshape(len(test_data_x), 784, 1)
    #     test_data_target = all_data.target[random_index2].astype(int)
    #     output_digit = []
    #     for i in range(10000):
    #         mnistNw.feed_forward_all_layers(test_data_x[i])
    #         output = mnistNw.all_layers[2].output_array
    #         output_digit.append(np.argmax(output))
    #     accuracy = sum(int(x == y) for (x, y) in tuple(zip(output_digit, test_data_target)))
    #     accuracy = accuracy / 10000
    #     print(accuracy)
