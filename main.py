import numpy as np
import os
from mlp import MLP
from trainer import MLPTrainer
from iris_utils import load_iris_dataset, evaluate_predictions

def run_iris_mode():
    X_train, y_train, X_test, y_test = load_iris_dataset("iris/iris.data")

    model_filename = "saved_network.json"
    load_existing = input(f"Czy wczytać sieć z pliku '{model_filename}'? (t/n): ").strip().lower() == "t"

    if load_existing and os.path.exists(model_filename):
        print("Wczytywanie sieci z pliku...")
        mlp, loss_history = MLP.load_from_file(model_filename)
    else:
        print("Tworzenie nowej sieci...")
        mlp = MLP(layer_sizes=[4, 6, 3], use_bias=True)
        loss_history = []

    trainer = MLPTrainer(mlp, X_test=X_test, y_test=y_test)
    trainer.loss_history = loss_history
    trainer.test_loss_history = []

    if not load_existing or input("Czy przeprowadzić trening sieci? (t/n): ").strip().lower() == "t":
        momentum = float(input("Podaj wartość momentum (0.0 jeśli nie chcesz go używać): ").strip())
        shuffle = input("Czy tasować dane w każdej epoce? (t/n): ").strip().lower() == "t"
        log_interval = int(input("Co ile epok zapisywać błąd? (np. 5): ").strip())

        print("Wybierz warunek stopu treningu:")
        print("1. Liczba epok")
        print("2. Poziom błędu")
        print("3. Oba warunki")
        stop_choice = input("Twój wybór (1/2/3): ").strip()

        epochs = 300
        min_loss = None

        if stop_choice == "1":
            epochs = int(input("Podaj liczbę epok: ").strip())
        elif stop_choice == "2":
            min_loss = float(input("Podaj minimalny błąd (np. 0.002): ").strip())
        elif stop_choice == "3":
            epochs = int(input("Podaj maksymalną liczbę epok: ").strip())
            min_loss = float(input("Podaj minimalny błąd (np. 0.002): ").strip())

        trainer.momentum = momentum

        trainer.train(
            X_train, y_train,
            epochs=epochs,
            shuffle=shuffle,
            min_loss=min_loss,
            log_interval=log_interval
        )


        save = input(f"Czy zapisać wytrenowaną sieć do '{model_filename}'? (t/n): ").strip().lower() == "t"
        if save:
            mlp.save_to_file(model_filename, loss_history=trainer.loss_history)
            print("Sieć została zapisana.")
    else:
        print("Trening pominięty — używana jest wczytana sieć.")

    if trainer.loss_history:
        trainer.plot_loss()

    y_pred = trainer.predict(X_test)
    trainer.log_test_details_json(X_test, y_test)
    print("Szczegóły testu zapisane do pliku: test_log.json")

    results = evaluate_predictions(y_test, y_pred)
    print("\nSkutecznosc nauki:")
    print(f"Poprawna klasyfikacja obiektow: {results['accuracy']} / {results['total']}")
    print("\nMacierz pomylek:")
    print(results["confusion_matrix"])
    print("\nPrecision / Recall / F-measure:")
    print(results["metrics_per_class"])


def run_autoencoder_mode():
    print("=== Tryb: Autoenkoder ===")

    # Dane wejściowe
    X = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    y = X.copy()

    mlp = MLP(layer_sizes=[4, 2, 4], use_bias=True)
    trainer = MLPTrainer(mlp, learning_rate=0.2, momentum=0.9, X_test=X, y_test=y)

    trainer.train(X, y, epochs=2500, shuffle=True, log_interval=100)


    trainer.plot_loss()

    y_pred = trainer.predict(X)
    print("\nWyniki autoenkodera:")
    for i in range(len(X)):
        print(f"Wejście: {X[i]}, Wyjście: {y_pred[i]}")

if __name__ == "__main__":
    print("Wybierz tryb:")
    print("1. Klasyfikacja Iris")
    print("2. Autoenkoder")
    mode = input("Twój wybór (1/2): ").strip()

    if mode == "2":
        run_autoencoder_mode()
    else:
        run_iris_mode()
