import numpy as np
import matplotlib.pyplot as plt
import json

class MLPTrainer:
    # Klasa odpowiedzialna za trening i testowanie sieci MLP
    def __init__(self, mlp, learning_rate=0.1, momentum=0.0):
        self.mlp = mlp  # Sieć neuronowa typu MLP
        self.learning_rate = learning_rate  # Współczynnik uczenia
        self.momentum = momentum  # Momentum - wpływ poprzednich zmian wag
        self.loss_history = []  # Historia błędów dla wykresu

    def train(self, X, y, epochs=100, shuffle=True, min_loss=None, log_interval=1):
        # Trening sieci neuronowej
        # shuffle - czy tasować dane w każdej epoce
        # min_loss - jeśli średni błąd spadnie poniżej tej wartości, zatrzymaj naukę
        # log_interval - co ile epok zapisywać błąd do pliku

        loss_log = []

        for epoch in range(epochs):
            total_loss = 0.0

            # Tworzymy kolejność indeksów i losowo tasujemy jeśli trzeba
            indices = np.arange(len(X))
            if shuffle:
                np.random.shuffle(indices)

            # Przechodzimy po danych treningowych
            for i in indices:
                output = self.mlp.forward(X[i])  # Propagacja w przód
                loss = np.mean((output - y[i]) ** 2)  # Średni błąd MSE
                total_loss += loss

                self.mlp.backward(y[i], self.learning_rate, self.momentum)  # Uczenie sieci (backprop)

            average_loss = total_loss / len(X)  # Błąd średni dla całej epoki
            self.loss_history.append(average_loss)

            # Logowanie błędu do listy co log_interval epok
            if epoch % log_interval == 0:
                loss_log.append({
                    "epoch": epoch + 1,
                    "average_loss": average_loss
                })

            print(f"Epoka nr {epoch + 1}/{epochs}: Sredni blad: {average_loss:.4f}")

            # Wczesne zatrzymanie, jeśli osiągnięto wymagany poziom błędu
            if min_loss is not None and average_loss < min_loss:
                print(f"Przerwano na epoce {epoch + 1} (blad < {min_loss})")
                break

        # Zapis do pliku JSON
        with open("loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=4)

        # self.plot_loss()

    def log_test_details_json(self, X, y, filename="test_log.json"):
        log_entries = []

        for i in range(len(X)):
            input_vector = X[i]
            expected = y[i]
            output = self.mlp.forward(input_vector)
            error_vector = output - expected
            mse = np.mean(error_vector ** 2)

            # Pobieranie warstw
            hidden_layer = self.mlp.layers[0]
            output_layer = self.mlp.layers[-1]

            # Dane neuronów
            hidden_outputs = hidden_layer.get_outputs().tolist()
            output_outputs = output_layer.get_outputs().tolist()

            hidden_weights = [
                {"weights": neuron.weights.tolist(), "bias": neuron.bias}
                for neuron in hidden_layer.neurons
            ]
            output_weights = [
                {"weights": neuron.weights.tolist(), "bias": neuron.bias}
                for neuron in output_layer.neurons
            ]

            entry = {
                "input": input_vector.tolist(),
                "expected": expected.tolist(),
                "output": output.tolist(),
                "error": error_vector.tolist(),
                "mse": mse,
                "hidden_outputs": hidden_outputs,
                "output_outputs": output_outputs,
                "hidden_weights": hidden_weights,
                "output_weights": output_weights
            }

            log_entries.append(entry)

        with open(filename, "w") as f:
            json.dump(log_entries, f, indent=4)

    def predict(self, X):
        # Przewidywanie wyników (forward-only, bez uczenia)
        return np.array([self.mlp.forward(x) for x in X])

    def plot_loss(self):
        # Wykres błędu uczącej się sieci
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history)
        plt.xlabel("Numer epoki")
        plt.ylabel("Średni błąd")
        plt.title("Progres")
        plt.grid(True)
        plt.ylim(bottom=0)  # Oś Y od zera
        plt.tight_layout()
        plt.show()
