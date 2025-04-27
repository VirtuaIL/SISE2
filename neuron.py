import numpy as np

class Neuron:
    def __init__(self, num_inputs, use_bias=True):
        self.weights = np.random.uniform(-0.5, 0.5, num_inputs)

        if use_bias:
            self.bias = np.random.uniform(-0.5, 0.5)
        else:
            self.bias = 0.0

        self.use_bias = use_bias
        self.output = 0.0

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        total = np.dot(self.weights, inputs)

        if self.use_bias:
            total += self.bias

        self.output = self.activate(total)

        print(f"Neuron - wejścia: {inputs}, \nwagi: {self.weights}, \nbias: {self.bias},")
        print(f"suma ważona: {total}, \nwyjście po sigmoidzie: {self.output}")

        return self.output
