import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, use_bias=True):
        self.neurons = []

        for _ in range(num_neurons):
            neuron = Neuron(num_inputs_per_neuron, use_bias)
            self.neurons.append(neuron)

    def forward(self, inputs):
        outputs = []

        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)

        print(50*"-")
        print(f"Warstwa - wejścia: {inputs}")
        print(f"Warstwa - wyjścia: {outputs}")


        return np.array(outputs)
