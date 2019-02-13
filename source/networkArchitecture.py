import numpy as np

from source.connection import Connection
from source.layer import Layer
from source.loss import CrossEntropy


class NetworkArchitecture:
    def __init__(self, layers: [Layer], connection_indices: [tuple]):
        self.number_of_layers = len(layers)
        self.all_layers = layers
        self.connection_indices = connection_indices
        self.connections = self.create_connections()
        self.set_predecessors_and_successors()
        self.weights = [
            np.zeros([self.all_layers[to_layer_index].shape, self.all_layers[from_layer_index].shape], dtype=float)
            for from_layer_index, to_layer_index in self.connection_indices]
        self.feed_forward_sequence = self.get_feed_forward_sequence()
        self.back_propagation_sequence = list(reversed(self.feed_forward_sequence))
        self.loss = CrossEntropy()

    def create_connections(self):
        self.connections = []
        for connection in self.connection_indices:
            self.connections.append(Connection(self.all_layers[connection[0]], self.all_layers[connection[1]]))

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

    def feed_forward_all_layers(self, input_vectors):
        self.all_layers[0].activated_output = input_vectors
        for current_layer_index in self.feed_forward_sequence[1:]:
            current_input_vector = self.get_input_vectors_of_a_layer(current_layer_index)
            input_weights = self.get_input_weights_of_a_layer(current_layer_index)
            self.all_layers[current_layer_index].set_input_array(current_input_vector, input_weights)
            self.all_layers[current_layer_index].set_output_array()

    def get_input_weights_of_a_layer(self, current_layer_index: int):
        required_weight_indices = [self.connection_indices.index((predecessor, current_layer_index))
                                   for predecessor in self.all_layers[current_layer_index].predecessors]
        all_weights = [self.weights[weight_index] for weight_index in required_weight_indices]
        all_weights = np.concatenate(all_weights, axis=1)
        return all_weights

    def get_input_vectors_of_a_layer(self, current_layer_index: int) -> [np.array]:
        input_vectors = [self.all_layers[predecessor].activated_output
                         for predecessor in self.all_layers[current_layer_index].predecessors]
        highest_axis = input_vectors[0].ndim - 1
        input_vectors = np.concatenate(input_vectors, axis=highest_axis)
        return input_vectors

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
        for layer_index in range(self.number_of_layers):
            self.all_layers[layer_index].bias -= (eta / len(training_subset)) * gradient_for_biases[layer_index]
        for weight_index in range(len(self.connection_indices)):
            self.weights[weight_index] -= (eta / len(training_subset)) * gradient_for_weights[weight_index]

    def back_propagate_all_layers(self, training_data):
        gradient_for_biases = [np.zeros([1, self.all_layers[layer_index].shape]) for layer_index in
                               range(self.number_of_layers)]  # TODO: shape should be more dynamic
        gradient_for_weights = [np.zeros(w.shape) for w in self.weights]

        input_vectors = [x[0] for x in training_data]
        input_vectors = np.concatenate(input_vectors, axis=0)
        actual_y_vectors = [y[1] for y in training_data]
        actual_y_vectors = np.concatenate(actual_y_vectors, axis=0)
        self.update_deltas_of_all_layers(input_vectors, actual_y_vectors)

        for layer_index in range(self.number_of_layers):
            delta_sum = self.all_layers[layer_index].delta.sum(axis=0)
            delta_sum = delta_sum.reshape(np.append(1, delta_sum.shape))
            gradient_for_biases[layer_index] += delta_sum
        for weight_index in range(len(self.connection_indices)):
            predecessor, successor = self.connection_indices[weight_index]
            gradient_for_weights[weight_index] += np.dot(self.all_layers[successor].delta.transpose(),
                                                         self.all_layers[predecessor].activated_output)
        return gradient_for_biases, gradient_for_weights

    def update_deltas_of_all_layers(self, input_vectors, actual_y_vectors):
        self.feed_forward_all_layers(input_vectors)
        output_layer_index = self.back_propagation_sequence[0]
        predicted_y_vectors = self.all_layers[output_layer_index].activated_output
        self.all_layers[output_layer_index].delta = self.loss.get_delta_last_layer(predicted_y_vectors,
                                                                                   actual_y_vectors)
        for layer_index in self.back_propagation_sequence[1:]:
            successors_deltas = self.get_successor_deltas_of_a_layer(layer_index)
            output_weights = self.get_output_weights_of_a_layer(layer_index)
            self.all_layers[layer_index].set_delta(successors_deltas, output_weights)

    def get_output_weights_of_a_layer(self, current_layer_index: int):
        required_weight_indices = [self.connection_indices.index((current_layer_index, successor))
                                   for successor in self.all_layers[current_layer_index].successors]
        all_weights = [self.weights[weight_index] for weight_index in required_weight_indices]
        all_weights = np.concatenate(all_weights, axis=0)
        return all_weights

    def get_successor_deltas_of_a_layer(self, current_layer_index: int):
        delta_inputs = [self.all_layers[successor].delta
                        for successor in self.all_layers[current_layer_index].successors]
        highest_axis = delta_inputs[0].ndim - 1
        delta_inputs = np.concatenate(delta_inputs, axis=highest_axis)
        return delta_inputs
