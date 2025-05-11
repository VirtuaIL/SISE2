import numpy as np

class Neuron:

    def __init__(self, num_inputs, use_bias=True):
        self.weights = np.random.uniform(-0.5, 0.5, num_inputs)  # losowe wagi
        self.prev_weight_updates = np.zeros(num_inputs)  # poprzednie zmiany wag do momentum

        if use_bias:
            self.bias = np.random.uniform(-0.5, 0.5)
            self.prev_bias_update = 0.0
        else:
            self.bias = 0.0
            self.prev_bias_update = 0.0

        self.use_bias = use_bias
        self.output = 0.0 # Wyjście neuronu (po aktywacji)
        self.input = None # Wejście, które zapamiętujemy do obliczeń gradientu
        self.delta = 0.0  # Wartość błędu (gradientu) propagowanego wstecz

    def activate(self, x):
        # Sigmoidalna funkcja aktywacji
        return 1 / (1 + np.exp(-x))

    def activate_derivative(self):
        # Pochodna sigmoidy, potrzebna do obliczeń gradientu
        return self.output * (1 - self.output)

    def forward(self, inputs):
        # Obliczam wyjście neuronu
        self.input = inputs
        total = np.dot(self.weights, inputs)
        if self.use_bias:
            total += self.bias
        self.output = self.activate(total)
        return self.output

    def calculate_output_delta(self, target):
        # Oblicza deltę błędu dla warstwy wyjściowej
        self.delta = (target - self.output) * self.activate_derivative()

    def calculate_hidden_delta(self, next_weights, next_deltas):
        # Oblicza deltę błędu dla neuronów ukrytych
        self.delta = self.activate_derivative() * np.dot(next_weights, next_deltas)

    def update_weights(self, learning_rate, momentum):
        # Aktualizacja wag z użyciem momentu
        weight_update = learning_rate * self.delta * self.input + momentum * self.prev_weight_updates
        self.weights += weight_update
        self.prev_weight_updates = weight_update

        if self.use_bias:
            bias_update = learning_rate * self.delta + momentum * self.prev_bias_update
            self.bias += bias_update
            self.prev_bias_update = bias_update
