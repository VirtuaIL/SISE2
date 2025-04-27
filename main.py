import numpy as np
from mlp import MLP

if __name__ == "__main__":
    mlp = MLP(layer_sizes=[3, 3, 3, 2], use_bias=True)

    # 3 neurony wejściowe
    # 3 neurony wewnętrzne - 2 warstwy
    # 2 neurony wyjściowe

    # Generowanie losowego wejścia
    random_input = np.random.rand(3)

    print(f"Wejście do sieci: {random_input}")
    print(50*"=")
    output = mlp.forward(random_input)
    # print(50 * "=")
    print(f"Wyjście z sieci: {output}")
