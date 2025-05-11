import numpy as np
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Zamiana etykiety klasy na wektor one-hot
def label_to_one_hot(name):
    if name == 'Iris-setosa':
        return [1, 0, 0]
    elif name == 'Iris-versicolor':
        return [0, 1, 0]
    elif name == 'Iris-virginica':
        return [0, 0, 1]
    else:
        raise ValueError(f"Unknown label: {name}")

# Wczytanie i podział na zbiory
def load_iris_dataset(path="iris.data", test_ratio=0.2, seed=42):
    with open(path, "r") as file:
        lines = []
        for line in file:
            line = line.strip()
            if line:
                lines.append(line)

    # Losowe tasowanie danych
    random.seed(seed)
    random.shuffle(lines)

    data = []
    labels = []
    for line in lines:
        parts = line.split(',')
        features = list(map(float, parts[:4])) # 4 cechy wejściowe
        label = parts[4] # klasa
        data.append(features)
        labels.append(label_to_one_hot(label))

    data = np.array(data)
    labels = np.array(labels)

    # Podział danych na trening i testy
    split_index = int(len(data) * (1 - test_ratio))
    X_train = data[:split_index] # wszystko aż do indexu
    X_test = data[split_index:]
    y_train = labels[:split_index]
    y_test = labels[split_index:]

    return X_train, y_train, X_test, y_test

# Funkcja do oceny skuteczności klasyfikacji
def evaluate_predictions(y_true, y_pred):

    # zamieniam na wektory z indexami klas
    true_classes = np.argmax(y_true, axis=1)       # rzeczywiste etykiety
    predicted_classes = np.argmax(y_pred, axis=1)  # przewidywane etykiety <- wyjścia z sieci


    # tworze wektor z boolem w środku i zliczam true jako correct
    # true_classes == predicted_classes <- wektor z boolami
    # correct <- liczba wystąpień true
    correct = np.sum(true_classes == predicted_classes)

    # liczba testowych danych
    total = len(true_classes)

    matrix = confusion_matrix(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average=None, zero_division=0)
    recall = recall_score(true_classes, predicted_classes, average=None, zero_division=0)
    f1 = f1_score(true_classes, predicted_classes, average=None, zero_division=0)

    metrics = pd.DataFrame({
        "Class": ["Setosa", "Versicolor", "Virginica"],
        "Precision": precision,
        "Recall": recall,
        "F-measure": f1
    })

    return {
        "accuracy": correct,
        "total": total,
        "confusion_matrix": matrix,
        "metrics_per_class": metrics
    }
