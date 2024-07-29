import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split

class DataGenerator:
    @staticmethod
    def eval_func(x):
        return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

    @staticmethod
    def generate_data(start, end, num_points):
        X = np.linspace(start, end, num_points).reshape(-1, 1)
        y = DataGenerator.eval_func(X)
        return X, y

    def split_data(self, X, y, train_size=0.8):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        return X_train, X_test, y_train, y_test

    def plot_data(self, X, y, predictions):
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
    X, y = DataGenerator.generate_data(-10, 10, 1000)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(X, y)
    plt.title('Plot of the function')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
