import numpy as np
from data_generator import DataGenerator
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from copy import deepcopy

from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy
from typing import List
import pickle
import seaborn as sns
import pandas as pd

@dataclass
class NeuralNetwork:
    X: np.ndarray
    y: np.ndarray
    neurons: List[int]
    learning_rate: float
    clip_value: float
    weights: List[np.ndarray] = field(default_factory=list)
    biases: List[np.ndarray] = field(default_factory=list)
    layers: List[np.ndarray] = field(default_factory=list)
    min_val_loss: float = float("inf")
    best_weights: List[np.ndarray] = field(default_factory=list)
    best_biases: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.initialize_weights()

    def initialize_weights(self):
        input_size = self.X.shape[1]
        self.weights.append(
            np.random.randn(input_size, self.neurons[0]) * np.sqrt(2.0 / input_size)
        )
        self.biases.append(np.zeros((1, self.neurons[0])))

        for i in range(1, len(self.neurons)):
            self.weights.append(
                np.random.randn(self.neurons[i - 1], self.neurons[i])
                * np.sqrt(2.0 / self.neurons[i - 1])
            )
            self.biases.append(np.zeros((1, self.neurons[i])))

        self.weights.append(
            np.random.randn(self.neurons[-1], 1) * np.sqrt(2.0 / self.neurons[-1])
        )
        self.biases.append(np.zeros((1, 1)))

        self.layers = [None] * (len(self.neurons) + 1)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def feedforward(self, X):
        self.layers[0] = self.relu(np.dot(X, self.weights[0]) + self.biases[0])
        for i in range(1, len(self.neurons)):
            self.layers[i] = self.relu(
                np.dot(self.layers[i - 1], self.weights[i]) + self.biases[i]
            )
        self.layers[-1] = (
            np.dot(self.layers[-2], self.weights[-1]) + self.biases[-1]
        )  # Linear activation for output layer
        return self.layers[-1]

    def clip_gradients(self, gradients):
        return [np.clip(grad, -self.clip_value, self.clip_value) for grad in gradients]

    def set_params(self, params):
        start = 0
        end = 0

        for i in range(len(self.weights)):
            end += self.weights[i].size
            self.weights[i] = np.array(params[start:end]).reshape(self.weights[i].shape)
            start = end

        for i in range(len(self.biases)):
            end += self.biases[i].size
            self.biases[i] = np.array(params[start:end]).reshape(self.biases[i].shape)
            start = end

        self.layers = [None] * (len(self.neurons) + 1)

        self.min_val_loss = float("inf")
        self.best_weights = deepcopy(self.weights)
        self.best_biases = deepcopy(self.biases)

    def backprop(self, X, y):
        output = self.feedforward(X)
        error = output - y
        d_output = error

        d_weights = []
        d_biases = []

        for i in reversed(range(len(self.neurons) + 1)):
            d_layer = d_output
            if i != len(self.neurons):
                d_layer = d_output * self.relu_derivative(self.layers[i])

            weight_update = (
                np.dot(self.layers[i - 1].T, d_layer)
                if i != 0
                else np.dot(X.T, d_layer)
            )
            bias_update = np.sum(d_layer, axis=0, keepdims=True)

            d_weights.append(weight_update)
            d_biases.append(bias_update)

            d_output = np.dot(d_layer, self.weights[i].T)

        d_weights = d_weights[::-1]
        d_biases = d_biases[::-1]

        d_weights = self.clip_gradients(d_weights)
        d_biases = self.clip_gradients(d_biases)

        for i in range(len(self.neurons) + 1):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def train(self, X_train, y_train, X_val, y_val, epochs, save_weights=False):
        loss_history = []
        for epoch in range(epochs):
            self.backprop(X_train, y_train)
            val_output = self.feedforward(X_val)
            val_loss = self.mse(y_val, val_output)
            loss_history.append(val_loss)
            if epoch % 100 == 99:
                print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_weights = deepcopy(self.weights)
                self.best_biases = deepcopy(self.biases)

        if save_weights:
            self.save_weights("best_weights")

        return loss_history
    def use_best_weights(self):
        self.weights = deepcopy(self.best_weights)
        self.biases = deepcopy(self.best_biases)

    def predict(self, X):
        return self.feedforward(X)

    def save_weights(self, filename):
        with open("weights/" + filename + "_weights.pkl", "wb") as f:
            pickle.dump(self.best_weights, f)
        with open("weights/" + filename + "_biases.pkl", "wb") as f:
            pickle.dump(self.best_biases, f)

    def load_weights(self, filename):
        with open("weights/" + filename + "_weights.pkl", "rb") as f:
            self.best_weights = pickle.load(f)
            print(np.array(self.best_weights)[0].shape)
        with open("weights/" + filename + "_biases.pkl", "rb") as f:
            self.best_biases = pickle.load(f)

    def get_params(self):
        params = np.concatenate(
            [w.flatten() for w in self.weights] + [b.flatten() for b in self.biases]
        )
        return params.tolist()

    def set_params(self, params):
        start = 0
        end = 0

        for i in range(len(self.weights)):
            end += self.weights[i].size
            self.weights[i] = np.array(params[start:end]).reshape(self.weights[i].shape)
            start = end

        for i in range(len(self.biases)):
            end += self.biases[i].size
            self.biases[i] = np.array(params[start:end]).reshape(self.biases[i].shape)
            start = end

        self.layers = [None] * (len(self.neurons) + 1)

        self.min_val_loss = float("inf")
        self.best_weights = deepcopy(self.weights)
        self.best_biases = deepcopy(self.biases)


def train_nn(config):
    data_generator = DataGenerator()
    eval_func = data_generator.eval_func
    X, y = data_generator.generate_data(-10, 10, 500)

    nn = NeuralNetwork(
        X,
        y,
        neurons=config,
        learning_rate=0.0001,
        clip_value=1.0,
    )
    nn.load_weights("best_weights")
    nn.use_best_weights()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    loss_history = nn.train(X_train, y_train, X_test, y_test, 100_000, save_weights=True)
    y_pred = nn.predict(X_test)

    test_loss = nn.mse(y_test, y_pred)
    return test_loss, loss_history


def load_and_predict():
    data_generator = DataGenerator()
    eval_func = data_generator.eval_func
    X, y = data_generator.generate_data(-10, 10, 500)

    nn = NeuralNetwork(
        X,
        y,
        neurons=[200, 100, 100, 50],
        learning_rate=0.0001,
        clip_value=1.0,
    )
    nn.load_weights("best")
    nn.use_best_weights()

    X_range = np.linspace(-10, 10, 1000).reshape(-1, 1)

    y_pred = nn.predict(X_range)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=X_range.squeeze(), y=y_pred.squeeze(), label="Predicted")
    sns.lineplot(x=X.squeeze(), y=eval_func(X).squeeze(), label="True")
    plt.title("Comparison of predicted and true function")
    plt.xlabel("X")
    plt.ylabel("y")

    plt.legend()
    plt.grid(True)
    plt.savefig("comparison.png", dpi=600)
    plt.show()


def main():

    # neuron_configs = [[50, 25, 10], [100, 50, 25], [200, 100, 50]]

    # results = []

    # for config in neuron_configs:

    #     test_loss = train_nn(config)

    #     results.append({
    #         'Configuration': str(config),
    #         'Test Loss': test_loss,
    #     })

    # df = pd.DataFrame(results)

    # print(df)
    config = [100, 50, 25]

    test_loss, loss_history = train_nn(config)

    print(f'Configuration: {config}, Test Loss: {test_loss}')

    window = np.ones(100) / 100


    averaged_loss_history = np.convolve(loss_history, window, 'valid')

    plt.plot(averaged_loss_history, label=str(config))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
