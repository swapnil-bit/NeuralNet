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

    def feed_forward_all_layers(self, input_arrays):
        self.all_layers[0].output_array = input_arrays
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

    def train_netwrok(self, training_data, epochs: int, batch_size: int, eta: float):
        if batch_size == 0:
            batch_size = len(training_data)
        number_of_batches = int(np.floor(len(training_data) / batch_size))
        for iteration in range(epochs):
            random_index = np.random.choice(len(training_data), len(training_data), replace=False)
            shuffled_training_data = [training_data[index] for index in random_index]
            for batch_index in range(number_of_batches):
                training_subset = shuffled_training_data[(batch_index * batch_size):((batch_index + 1) * batch_size)]
                self.train_network_for_single_batch(training_subset, eta)

    def train_network_for_single_batch(self, training_subset, eta):
        gradient_for_biases, gradient_for_weights = self.back_propagate_all_layers(training_subset)
        # TODO: these updates should be moved to layer and connection classes
        for layer in self.all_layers.values():
            layer.bias -= (eta / len(training_subset)) * gradient_for_biases[layer.id]
        for connection in self.connections.values():
            connection.weights -= (eta / len(training_subset)) * gradient_for_weights[connection.id]

    def back_propagate_all_layers(self, training_data):
        # TODO: shapes of biases should be more dynamic to accomodate multi dimensional array
        gradient_for_biases = dict((layer.id, np.zeros(layer.shape)) for layer in self.all_layers.values())
        gradient_for_weights = dict((connection.id, np.zeros(connection.weights.shape)) for connection in self.connections.values())

        input_arrays = [x[0] for x in training_data]
        input_arrays = np.concatenate(input_arrays, axis=0)  # TODO: This works only with batch size 1
        actual_y_arrays = [y[1] for y in training_data]
        actual_y_arrays = np.concatenate(actual_y_arrays, axis=0)  # TODO: This works only with batch size 1
        self.update_deltas_of_all_layers(input_arrays, actual_y_arrays)

        for layer in self.all_layers.values():
            delta_sum = layer.delta.sum(axis=0)
            # delta_sum = delta_sum.reshape(np.append(1, delta_sum.shape))  # TODO: is this shape dynamic?
            gradient_for_biases[layer.id] += delta_sum
        for connection in self.connections.values():
            predecessor, successor = connection.id
            # TODO: Below lines are gross deviation from actual code. It will work for only 1D arrays.
            # TODO: Actually it should be tensordot product with right axis length without any reshape.
            # TODO: Shape of tensordot should match weights shape.
            required_shape = connection.weights.shape
            gradient_for_weights[connection.id] += (np.kron(self.all_layers[successor].delta,
                                                           self.all_layers[predecessor].output_array)).reshape(required_shape)
        return gradient_for_biases, gradient_for_weights

    def update_deltas_of_all_layers(self, input_arrays, actual_y_arrays):
        self.feed_forward_all_layers(input_arrays)
        output_layer_id = self.back_propagation_sequence[0]
        predicted_y_arrays = self.all_layers[output_layer_id].output_array
        self.all_layers[output_layer_id].delta = self.loss.get_delta_last_layer(predicted_y_arrays, actual_y_arrays)
        for layer_id in self.back_propagation_sequence[1:]:
            successors_deltas = self.get_successor_deltas_of_a_layer(layer_id)
            output_weights = self.get_output_weights_of_a_layer(layer_id)
            self.all_layers[layer_id].set_delta(successors_deltas, output_weights)

    def get_output_weights_of_a_layer(self, current_layer_id: int):
        all_weights = [self.connections[(current_layer_id, successor)].weights for successor
                       in self.all_layers[current_layer_id].successors]
        all_weights = np.concatenate(all_weights, axis=0)
        return all_weights

    def get_successor_deltas_of_a_layer(self, current_layer_id: int):
        delta_inputs = [self.all_layers[successor].delta
                        for successor in self.all_layers[current_layer_id].successors]
        highest_axis = delta_inputs[0].ndim - 1
        delta_inputs = np.concatenate(delta_inputs, axis=highest_axis)
        return delta_inputs
