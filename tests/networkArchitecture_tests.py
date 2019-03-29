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

    def testThat_getInputArrays_givesArraysInCorrectDimensions_ForMultipleInputsInSingleBatchSize(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        network1.all_layers[2].output_array = [np.zeros([2])]
        network1.all_layers[3].output_array = [np.zeros([3])]
        expected_arrays = [[np.zeros([3])], [np.zeros([3])]]
        actual_arrays = network1.get_input_arrays_of_a_layer(4)
        self.assertTrue((expected_arrays[0][0] == actual_arrays[0][0]).all())
        self.assertTrue((expected_arrays[1][0] == actual_arrays[1][0]).all())

    def testThat_getInputVector_givesVectorsInCorrectDimensions_ForMultipleInputsInMultipleBatchSize(self):
        batch_size = 4
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 3)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        network1.all_layers[1].output_array = [np.zeros([3])] * batch_size
        network1.all_layers[2].output_array = [np.ones([2])] * batch_size
        network1.connections[(1, 3)].weights = np.ones([3, 3])
        network1.connections[(2, 3)].weights = np.ones([3, 2])
        expected_vectors = [[np.zeros([3])] * batch_size, [np.ones([3]) * 2] * batch_size]
        actual_vectors = network1.get_input_arrays_of_a_layer(3)
        for i in range(2):
            for j in range(batch_size):
                self.assertTrue((expected_vectors[i][j] == actual_vectors[i][j]).all())

    def testThat_feedForwardAllLayers_givesOutputInCorrectDimensions_forSingleBatchSize(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 4])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        expected_last_layer_output = [np.array([0.5] * 4)]
        network1.feed_forward_all_layers([np.array([0, 0, 0])])
        self.assertTrue((expected_last_layer_output[0] == network1.all_layers[4].output_array[0]).all())

    def testThat_feedForwardAllLayers_givesOutputInCorrectDimensions_forMultipleBatchSize(self):
        batch_size = 4
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 4])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        network1 = NetworkArchitecture(layers, connections)
        expected_last_layer_output = [np.array([0.5] * 4)] * batch_size
        network1.feed_forward_all_layers([np.array([0, 0, 0])] * batch_size)
        for i in range(batch_size):
            self.assertTrue((expected_last_layer_output[i] == network1.all_layers[4].output_array[i]).all())

    def testThat_updateDeltaOfLastLayer_givesCorrectLastLayerDelta_forSingleBatchSize(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        input_arrays = [np.array([1, 2, 3])]
        actual_y_arrays = [np.array([0, 1, 0])]
        expected_deltas = [np.array([0.5, -0.5, 0.5])]
        network1 = NetworkArchitecture(layers, connections)
        network1.update_delta_of_last_layer(actual_y_arrays, input_arrays)
        actual_deltas = network1.all_layers[4].delta
        self.assertTrue((expected_deltas[0] == actual_deltas[0]).all())

    def testThat_updateDeltaOfLastLayer_givesCorrectLastLayerDeltas_forMultipleBatchSize(self):
        batch_size = 3
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        input_arrays = [np.array([1, 2, 3])] * batch_size
        actual_y_arrays = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        expected_deltas = [np.array([-0.5, 0.5, 0.5]), np.array([0.5, -0.5, 0.5]), np.array([0.5, 0.5, -0.5])]
        network1 = NetworkArchitecture(layers, connections)
        network1.update_delta_of_last_layer(actual_y_arrays, input_arrays)
        actual_deltas = network1.all_layers[4].delta
        for i in range(batch_size):
            self.assertTrue((expected_deltas[i] == actual_deltas[i]).all())

    def testThat_updateDeltaOfAllLayers_flowsCorrectly_withMultipleSuccessorAndSingleBatchSize(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.array([1, 2, 3])]
        actual_y_arrays = [np.array([0, 1, 0])]
        expected_deltas = [np.array([0.0023, 0.0012, 0.0005])]
        network1 = NetworkArchitecture(layers, connections)
        network1.update_deltas_of_all_layers(input_arrays, actual_y_arrays, 1)
        self.assertTrue((expected_deltas[0] == np.around(network1.all_layers[0].delta[0], 4)).all())

    def testThat_updateDeltaOfAllLayers_flowsCorrectly_withMultipleSuccessorAndMultipleBatchSize(self):
        batch_size = 4
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.array([1, 2, 3])] * batch_size
        actual_y_arrays = [np.array([0, 1, 0])] * batch_size
        expected_deltas = [np.array([0.0023, 0.0012, 0.0005])] * batch_size
        network1 = NetworkArchitecture(layers, connections)
        network1.update_deltas_of_all_layers(input_arrays, actual_y_arrays, batch_size)
        for i in range(batch_size):
            self.assertTrue((expected_deltas[i] == np.around(network1.all_layers[0].delta[i], 4)).all())

    def testThat_updateDeltaOfAllLayers_flowsCorrectly_withSingleInputSingleBatchSizeAndSingleNeuronInLastLayer(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 1])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.zeros([3])]
        actual_y_arrays = [np.ones([1])]
        expected_delta = [np.array([-0.0064, -0.0064, -0.0064])]
        network1 = NetworkArchitecture(layers, connections)
        network1.update_deltas_of_all_layers(input_arrays, actual_y_arrays, 1)
        self.assertTrue((expected_delta[0] == np.round(network1.all_layers[0].delta[0], 4)).all())

    def testThat_updateDeltaOfAllLayers_flowsCorrectly_withMultipleInputSingleBatchSizeAndSingleNeuronInLastLayer(self):
        batch_size = 4
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 1])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.zeros([3])] * batch_size
        actual_y_arrays = [np.ones([1])] * batch_size
        expected_delta = [np.array([-0.0064, -0.0064, -0.0064])] * batch_size
        network1 = NetworkArchitecture(layers, connections)
        network1.update_deltas_of_all_layers(input_arrays, actual_y_arrays, batch_size)
        for i in range(batch_size):
            self.assertTrue((expected_delta[i] == np.round(network1.all_layers[0].delta[i], 4)).all())

    def testThat_backPropagate_flowsCorrectly_withSingleBatchSize(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.zeros([3])]
        actual_y_arrays = [np.array([0, 1, 0])]
        training_data = [(input_arrays[0], actual_y_arrays[0])]
        expected_gradient_for_biases = np.array([0.3986, 0.3986, 0.3986])
        expected_gradient_for_weights = np.array([[0, 0, 0], [0, 0, 0]])
        network1 = NetworkArchitecture(layers, connections)
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data, 1)
        self.assertTrue((expected_gradient_for_biases == np.round(gradient_for_biases[0], 4)).all())
        self.assertTrue((expected_gradient_for_weights == np.round(gradient_for_weights[(0, 2)], 4)).all())

    def testThat_backPropagate_flowsCorrectly_withMultipleBatchSize(self):
        batch_size = 4
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 3, 2, 3, 3])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.zeros([3])] * batch_size
        actual_y_arrays = [np.array([0, 1, 0])] * batch_size
        training_data = [(input_arrays[i], actual_y_arrays[i]) for i in range(batch_size)]
        expected_gradient_for_biases = np.array([1.5942, 1.5942, 1.5942])
        expected_gradient_for_weights = np.array([[0, 0, 0], [0, 0, 0]])
        network1 = NetworkArchitecture(layers, connections)
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data, batch_size)
        self.assertTrue((expected_gradient_for_biases == np.round(gradient_for_biases[0], 4)).all())
        self.assertTrue((expected_gradient_for_weights == np.round(gradient_for_weights[(0, 2)], 4)).all())

    def testThat_backPropagate_flowsCorrectly_withSingleBatchSizeAndSingleNeuronInLastLayer(self):
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 1, 2, 3, 1])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.zeros([3])]
        actual_y_arrays = [np.ones([1])]
        training_data = [(input_arrays[0], actual_y_arrays[0])]
        expected_gradient_for_biases = np.array([-0.0091, -0.0091, -0.0091])
        expected_gradient_for_weights = np.array([[0, 0, 0], [0, 0, 0]])
        network1 = NetworkArchitecture(layers, connections)
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data, 1)
        self.assertTrue((expected_gradient_for_biases == np.round(gradient_for_biases[0], 4)).all())
        self.assertTrue((expected_gradient_for_weights == np.round(gradient_for_weights[(0, 2)], 4)).all())

    def testThat_backPropagate_flowsCorrectly_withMultipleBatchSizeAndSingleNeuronInLastLayer(self):
        batch_size = 4
        layers = self.create_list_of_1D_layers_with_sigmoid_activations([3, 1, 2, 3, 1])
        connection_indices = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices)
        for index in range(len(connection_indices)):
            connections[index].weights = np.ones(connections[index].weights.shape)
        input_arrays = [np.zeros([3])] * batch_size
        actual_y_arrays = [np.ones([1])] * batch_size
        training_data = [(input_arrays[i], actual_y_arrays[i]) for i in range(batch_size)]
        expected_gradient_for_biases = np.array([-0.0364, -0.0364, -0.0364])
        expected_gradient_for_weights = np.array([[0, 0, 0], [0, 0, 0]])
        network1 = NetworkArchitecture(layers, connections)
        gradient_for_biases, gradient_for_weights = network1.back_propagate_all_layers(training_data, batch_size)
        self.assertTrue((expected_gradient_for_biases == np.round(gradient_for_biases[0], 4)).all())
        self.assertTrue((expected_gradient_for_weights == np.round(gradient_for_weights[(0, 2)], 4)).all())

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
    #     batch_size = 25
    #     layers = self.create_list_of_1D_layers_with_sigmoid_activations([784, 100, 10])
    #     connection_indices = [(0, 1), (1, 2)]
    #     connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices, "normal")
    #     mnistNw = NetworkArchitecture(layers, connections)
    #     mnistNw.train_network(training_data, epochs, batch_size, learning_rate)
    #
    #     ## Prediction on test data
    #     test_data_x = all_data.data[random_index2] / 25500
    #     test_data_x = test_data_x.reshape(len(test_data_x), 784)
    #     test_data_target = all_data.target[random_index2].astype(int)
    #     output_digit = []
    #     for i in range(10000):
    #         mnistNw.feed_forward_all_layers([test_data_x[i]])
    #         output = mnistNw.all_layers[2].output_array[0]
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
    #     batch_size = 25
    #     layers = self.create_list_of_1D_layers_with_2D_structures_and_sigmoid_activations([784, 100, 10])
    #     connection_indices = [(0, 1), (1, 2)]
    #     connections = self.create_list_of_connections_having_fully_connection_type(layers, connection_indices, "normal")
    #     mnistNw = NetworkArchitecture(layers, connections)
    #     mnistNw.train_network(training_data, epochs, batch_size, learning_rate)
    #
    #     ## Prediction on test data
    #     test_data_x = all_data.data[random_index2] / 25500
    #     test_data_x = test_data_x.reshape(len(test_data_x), 784, 1)
    #     test_data_target = all_data.target[random_index2].astype(int)
    #     output_digit = []
    #     for i in range(10000):
    #         mnistNw.feed_forward_all_layers([test_data_x[i]])
    #         output = mnistNw.all_layers[2].output_array[0]
    #         output_digit.append(np.argmax(output))
    #     accuracy = sum(int(x == y) for (x, y) in tuple(zip(output_digit, test_data_target)))
    #     accuracy = accuracy / 10000
    #     print(accuracy)
