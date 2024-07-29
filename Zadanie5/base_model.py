import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_generator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


print(
    "==========================================================================================="
)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def create_model(hidden_layers, neurons, learning_rate):
    # Define the model
    model = Sequential()

    # Add the hidden layers
    for _ in range(hidden_layers):
        model.add(Dense(neurons, activation="relu"))

    # Add the output layer
    model.add(Dense(1, activation="linear"))
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
    )

    return model


def plot_data(X, y, X_pred, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label="Actual")
    plt.plot(X, predictions, label="Predictions")
    plt.title("Plot of the function")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    hidden_layers = 3
    neurons = 100
    learning_rate = 0.001
    epochs = 1000
    gen = DataGenerator()
    X, y = gen.generate_data(-10, 10, 500)

    X_train, X_test, y_train, y_test = gen.split_data(X, y)

    model = create_model(hidden_layers, neurons, learning_rate)

    checkpoint = ModelCheckpoint(
        "best_weights.h5", save_best_only=True, monitor="val_loss", mode="min"
    )

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[checkpoint])
    model.build(X_train.shape)
    # loss = model.evaluate(X_test, y_test)
    # print(f'Test loss: {loss}')

    model.load_weights("best_weights.h5")

    X_range = np.linspace(-10, 10, 1000).reshape(-1, 1)

    y_pred = model.predict(X_range)

    plt.figure(figsize=(10, 6))
    plt.plot(X_range, y_pred, label="Predicted")
    plt.plot(X_range, gen.eval_func(X_range), label="True")
    plt.title("Comparison of predicted and true function")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
