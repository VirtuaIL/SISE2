from layer import Layer
import numpy as np
import json

class MLP:
    def __init__(self, layer_sizes, use_bias=True):
        self.layers = []
        self.use_bias = use_bias
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(
                    num_neurons=layer_sizes[i + 1], # +1 bo nie tworze de facto tej pierwszej warstwy
                    num_inputs_per_neuron=layer_sizes[i],
                    use_bias=use_bias))

    def forward(self, inputs):
        # Przepuszcza dane przez całą sieć
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, expected_output, learning_rate, momentum):
        # Propagacja błędu wstecz przez sieć
        self.layers[-1].backward_output_layer(expected_output)
        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].backward_hidden_layer(self.layers[i + 1])
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def save_to_file(self, filename, loss_history=None):

        data = {
            "use_bias": self.use_bias,
            "layers": []
        }

        if loss_history is not None:
            data["loss_history"] = loss_history

        for layer in self.layers:
            layer_data = {
                "neurons": []
            }
            for neuron in layer.neurons:
                neuron_data = {
                    "weights": neuron.weights.tolist(),
                    "bias": neuron.bias
                }
                layer_data["neurons"].append(neuron_data)
            data["layers"].append(layer_data)
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "r") as file:
            data = json.load(file)

        use_bias = data.get("use_bias", True)
        layer_sizes = [len(data["layers"][0]["neurons"][0]["weights"])]
        for layer in data["layers"]:
            layer_sizes.append(len(layer["neurons"]))

        mlp = cls(layer_sizes, use_bias=use_bias)

        for i, layer_data in enumerate(data["layers"]):
            for j, neuron_data in enumerate(layer_data["neurons"]):
                mlp.layers[i].neurons[j].weights = np.array(neuron_data["weights"])
                mlp.layers[i].neurons[j].bias = neuron_data["bias"]

        loss_history = data.get("loss_history", [])
        return mlp, loss_history
