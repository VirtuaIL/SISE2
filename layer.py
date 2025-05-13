import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, use_bias=True):
        # Tworzy listę neuronów, każdy ma tyle wejść ile wyjść poprzedniej warstwy
        self.neurons = [Neuron(num_inputs_per_neuron, use_bias) for _ in range(num_neurons)]

    def forward(self, inputs):
        # Propagacja w przód: oblicza wyjścia wszystkich neuronów w warstwie
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def get_outputs(self):
        # Zwraca wyjścia wszystkich neuronów - debug
        return np.array([neuron.output for neuron in self.neurons])

    def backward_output_layer(self, expected_outputs):
        # Obliczanie delty błędów dla warstwy wyjściowej
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            expected = expected_outputs[i]
            neuron.calculate_output_delta(expected)

    def backward_hidden_layer(self, next_layer):
        # Oblicz delty błędów dla warstwy ukrytej na podstawie wag i delt warstwy kolejnej
        next_weights = np.array([n.weights for n in next_layer.neurons])  # macierz wag kolejnej warstwy - 2 wymiarowa
        next_deltas = np.array([n.delta for n in next_layer.neurons])     # delta kolejnej warstwy
        for i, neuron in enumerate(self.neurons):
            neuron.calculate_hidden_delta(next_weights[:, i], next_deltas)

    def update_weights(self, learning_rate, momentum):
        # Zaktualizuj wagi każdego neuronu w warstwie
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)
