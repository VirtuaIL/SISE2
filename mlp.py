from layer import Layer

class MLP:
    def __init__(self, layer_sizes, use_bias=True):
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            layer = Layer(output_size, input_size, use_bias)
            self.layers.append(layer)

    def forward(self, inputs):
        for idx, layer in enumerate(self.layers):
            print(f"                Warstwa nr {idx}: \nWarstwa - wejścia = {inputs}")
            inputs = layer.forward(inputs)
            print(50 * "=")

        return inputs
