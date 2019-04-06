import numpy as np

from source.connection import Connection
from source.layer import Layer
from source.loss import CrossEntropy


class NetworkArchitecture:
    def __init__(self, layers: [Layer], connections: [Connection]):
        self.number_of_layers = len(layers)
        self.all_layers = dict((layer.id, layer) for layer in layers)
        self.connections = dict((connection.id, connection) for connection in connections)
        self.layer_indices = list(self.all_layers.keys())
        self.connection_indices = list(self.connections.keys())
        self.set_predecessors_and_successors()
        self.feed_forward_sequence = self.get_feed_forward_sequence()
        self.back_propagation_sequence = list(reversed(self.feed_forward_sequence))
        self.loss = CrossEntropy()

    def set_predecessors_and_successors(self):
        for current_layer_index in range(self.number_of_layers):
            predecessor_list = [predecessor for (predecessor, successor) in self.connection_indices
                                if successor == current_layer_index]
            self.all_layers[current_layer_index].set_predecessor_list(predecessor_list)
            successor_list = [successor for (predecessor, successor) in self.connection_indices
                              if predecessor == current_layer_index]
            self.all_layers[current_layer_index].set_successor_list(successor_list)

    def get_feed_forward_sequence(self) -> [int]:
        unprocessed_layers = [x for x in range(self.number_of_layers)]
        processed_layers = list()
        while len(unprocessed_layers) > 0:
            for current_layer in unprocessed_layers:
                if set(self.all_layers[current_layer].predecessors).issubset(processed_layers):
                    processed_layers.append(current_layer)
                    unprocessed_layers.remove(current_layer)
                    break
        return processed_layers

    def feed_forward_all_layers(self, input_arrays: [np.array]):
        self.all_layers[0].input_array = [input.reshape(self.all_layers[0].shape) for input in input_arrays]
        self.all_layers[0].output_array = self.all_layers[0].input_array
        for layer_id in self.feed_forward_sequence[1:]:
            current_input_arrays = self.get_input_arrays_of_a_layer(layer_id)
            self.all_layers[layer_id].set_input_array(current_input_arrays)
            self.all_layers[layer_id].set_output_array()

    def get_input_arrays_of_a_layer(self, layer_id: int) -> [np.array]:
        input_arrays = list()
        for predecessor in self.all_layers[layer_id].predecessors:
            current_array = self.connections[(predecessor, layer_id)].transform_input(
                self.all_layers[predecessor].output_array)
            input_arrays.append(current_array)
        return input_arrays

    def train_network(self, training_data, epochs: int, batch_size: int, eta: float):
        if batch_size == 0:
            batch_size = len(training_data)
        number_of_batches = int(np.floor(len(training_data) / batch_size))
        for iteration in range(epochs):
            random_index = np.random.choice(len(training_data), len(training_data), replace=False)
            shuffled_training_data = [training_data[index] for index in random_index]
            for batch_index in range(number_of_batches):
                training_subset = shuffled_training_data[(batch_index * batch_size):((batch_index + 1) * batch_size)]
                self.train_network_for_single_batch(training_subset, eta, batch_size)

    def train_network_for_single_batch(self, training_subset, eta: float, batch_size: int):
        gradient_for_biases, gradient_for_weights = self.back_propagate_all_layers(training_subset, batch_size)
        # TODO: these updates should be moved to layer and connection classes
        for layer in self.all_layers.values():
            layer.bias -= (eta / len(training_subset)) * gradient_for_biases[layer.id]
        for connection in self.connections.values():
            connection.weights -= (eta / len(training_subset)) * gradient_for_weights[connection.id]

    def back_propagate_all_layers(self, training_subset, batch_size: int):
        gradient_for_biases = dict((layer.id, np.zeros(layer.shape)) for layer in self.all_layers.values())
        gradient_for_weights = dict(
            (connection.id, np.zeros(connection.weights.shape)) for connection in self.connections.values())

        input_arrays = [x[0] for x in training_subset]
        actual_y_arrays = [y[1] for y in training_subset]
        self.update_deltas_of_all_layers(input_arrays, actual_y_arrays, batch_size)

        for layer in self.all_layers.values():
            delta_sum = sum(layer.delta)
            gradient_for_biases[layer.id] += delta_sum
        for connection in self.connections.values():
            gradient_for_weights[connection.id] += connection.get_gradient_for_weights()
        return gradient_for_biases, gradient_for_weights

    def update_deltas_of_all_layers(self, input_arrays: [np.array], actual_y_arrays: [np.array], batch_size: int):
        self.update_delta_of_last_layer(actual_y_arrays, input_arrays)
        for layer_id in self.back_propagation_sequence[1:]:
            self.update_delta_of_single_layer(layer_id, batch_size)

    def update_delta_of_last_layer(self, actual_y_arrays, input_arrays):
        self.feed_forward_all_layers(input_arrays)
        output_layer_id = self.back_propagation_sequence[0]
        predicted_y_arrays = self.all_layers[output_layer_id].output_array
        self.all_layers[output_layer_id].delta = self.loss.get_delta_last_layer(predicted_y_arrays, actual_y_arrays)

    def update_delta_of_single_layer(self, layer_id: int, batch_size: int):
        current_layer_delta = [np.zeros(self.all_layers[layer_id].shape)] * batch_size
        for successor in self.all_layers[layer_id].successors:
            back_propagated_delta = self.connections[(layer_id, successor)].get_input_layer_delta()
            current_layer_delta = [(current_layer_delta[i] + back_propagated_delta[i]) for i in range(batch_size)]
        self.all_layers[layer_id].delta = current_layer_delta

