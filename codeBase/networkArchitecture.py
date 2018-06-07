import numpy as np
from codeBase.layer import Layer
from codeBase.networkConfigurations import NetworkConfigurations


class NetworkArchitecture:
    def __init__(self, layers: [Layer], layer_connections: [tuple]):
        self.number_of_layers = len(layers)
        self.all_layers = layers
        self.layer_connections = layer_connections
        self.set_predecessors_and_successors()
        self.weights = [
            np.zeros((self.all_layers[to_layer_index].size, self.all_layers[from_layer_index].size), dtype=float)
            for from_layer_index, to_layer_index in self.layer_connections]
        self.feed_forward_sequence = self.get_feed_forward_sequence()
        self.back_propagation_sequence = list(reversed(self.feed_forward_sequence))
        self.config = NetworkConfigurations()

    def set_predecessors_and_successors(self):
        for current_layer_index in range(self.number_of_layers):
            predecessor_list = [predecessor for (predecessor, successor) in self.layer_connections
                                if successor == current_layer_index]
            self.all_layers[current_layer_index].set_predecessor_list(predecessor_list)
            successor_list = [successor for (predecessor, successor) in self.layer_connections
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

    def get_input_weights_of_a_layer(self, current_layer_index: int):
        required_weight_indices = [self.layer_connections.index((predecessor, current_layer_index))
                                   for predecessor in self.all_layers[current_layer_index].predecessors]
        all_weights = [self.weights[weight_index] for weight_index in required_weight_indices]
        return all_weights

    def get_input_vectors_of_a_layer(self, current_layer_index: int) -> [np.array]:
        input_vectors = [self.all_layers[predecessor].activated_output
                         for predecessor in self.all_layers[current_layer_index].predecessors]
        return input_vectors

    def feed_forward_all_layers(self, input_vector):
        self.all_layers[0].activated_output = input_vector
        for current_layer_index in self.feed_forward_sequence[1:]:
            current_input_vector = self.get_input_vectors_of_a_layer(current_layer_index)
            input_weights = self.get_input_weights_of_a_layer(current_layer_index)
            self.all_layers[current_layer_index].set_linear_output(current_input_vector, input_weights)
            self.all_layers[current_layer_index].set_activated_output()

    def get_output_weights_of_a_layer(self, current_layer_index: int):
        required_weight_indices = [self.layer_connections.index((current_layer_index, successor))
                                   for successor in self.all_layers[current_layer_index].successors]
        all_weights = [self.weights[weight_index] for weight_index in required_weight_indices]
        return all_weights

    def get_successor_deltas_of_a_layer(self, current_layer_index: int):
        delta_inputs = [self.all_layers[successor].delta
                        for successor in self.all_layers[current_layer_index].successors]
        return delta_inputs

    def update_deltas_of_all_layers(self, input_vector, actual_y):
        self.feed_forward_all_layers(input_vector)
        output_layer_index = self.back_propagation_sequence[0]
        predicted_y = self.all_layers[output_layer_index].activated_output
        self.all_layers[output_layer_index].delta = self.config.get_delta_last_layer(predicted_y, actual_y)
        for layer_index in self.back_propagation_sequence[1:]:
            successors_deltas = self.get_successor_deltas_of_a_layer(layer_index)
            output_weights = self.get_output_weights_of_a_layer(layer_index)
            self.all_layers[layer_index].set_delta(successors_deltas, output_weights)

    def back_propagate_all_layers(self, training_data):
        gradient_for_biases = [np.zeros(self.all_layers[layer_index].bias.shape) for layer_index in
                               range(self.number_of_layers)]
        gradient_for_weights = [np.zeros(w.shape) for w in self.weights]
        for x, actual_y in training_data:
            self.update_deltas_of_all_layers(x, actual_y)
            for layer_index in range(self.number_of_layers):
                gradient_for_biases[layer_index] += self.all_layers[layer_index].delta
            for weight_index in range(len(self.layer_connections)):
                predecessor, successor = self.layer_connections[weight_index]
                gradient_for_weights[weight_index] += np.dot(self.all_layers[successor].delta,
                                                             self.all_layers[predecessor].activated_output.transpose())

        return gradient_for_biases, gradient_for_weights

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
        for weight_index in range(len(self.layer_connections)):
            self.weights[weight_index] -= (eta / len(training_subset)) * gradient_for_weights[weight_index]
