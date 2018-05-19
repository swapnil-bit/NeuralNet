import numpy as np
from codeBase.layer import Layer
from codeBase.networkConfigurations import NetworkConfigurations


class NetworkArchitecture:
    def __init__(self, layer_sizes: [int], layer_connections: [tuple]):
        self.number_of_layers = len(layer_sizes)
        self.layer_connections = layer_connections
        self.all_layers = self.create_all_layers(layer_sizes)
        self.weights = [np.zeros([layer_sizes[to_layer], layer_sizes[from_layer]], dtype=float)
                        for from_layer, to_layer in self.layer_connections]
        self.feed_forward_sequence = self.get_feed_forward_sequence()
        self.back_propagation_sequence = list(reversed(self.feed_forward_sequence))
        self.config = NetworkConfigurations()

    def create_all_layers(self, layer_sizes) -> [Layer]:
        all_layers = list()
        for current_layer_index in range(self.number_of_layers):
            predecessor_list = [predecessor for (predecessor, successor) in self.layer_connections
                                if successor == current_layer_index]
            successor_list = [successor for (predecessor, successor) in self.layer_connections
                              if predecessor == current_layer_index]
            layer = Layer(layer_sizes[current_layer_index], predecessor_list, successor_list)
            all_layers.append(layer)
        return all_layers

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

    def networkTraining(self, training_data, max_iteration, eta):
        for iteration in range(max_iteration):
            gradient_for_biases, gradient_for_weights = self.back_propagate_all_layers(training_data)
            for layer_index in range(self.number_of_layers):
                self.all_layers[layer_index].bias -= (eta/len(training_data))*gradient_for_biases[layer_index]
            for weight_index in range(len(self.layer_connections)):
                self.weights[weight_index] -= (eta/len(training_data))*gradient_for_weights[weight_index]
