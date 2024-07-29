import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

@dataclass
class SVM:
    w: float = field(default=0.0, init=False)
    b: float = field(default=0.0, init=False)
    learning_rate: float = 0.001
    lambda_param: float = 0.01
    iters: int = 1000
    kernel: callable = field(default=lambda x, y: np.dot(x, y), init=True)
    kernel_params: dict = field(default_factory=dict)

    def linear_kernel(self, x, y):
        c = self.kernel_params.get('c', 0)
        return np.dot(x, y) + c

    def polynomial_kernel(self, x, y):
        d = self.kernel_params.get('d', 3)
        r = self.kernel_params.get('r', 0)
        gamma = self.kernel_params.get('gamma', 1)
        return (gamma * np.dot(x, y) + r) ** d

    def rbf_kernel(self, x, y):
        gamma = self.kernel_params.get('gamma', 0.1)
        distance = np.linalg.norm(x-y)**2
        return np.exp(-gamma * distance)

    def sigmoid_kernel(self, x, y):
        gamma = self.kernel_params.get('gamma', 0.01)
        r = self.kernel_params.get('r', 1)
        print(np.tanh(gamma * np.dot(x, y) + r))
        return np.tanh(gamma * np.dot(x, y) + r)

    def initialize_weights(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0

    def update_weights(self, class_labels, X):
        X = X.to_numpy()
        for iter in range(self.iters):
            for idx, x_i in enumerate(X):
                condition = class_labels[idx] * (self.kernel(self, x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, class_labels[idx]))
                    self.b -= self.learning_rate * class_labels[idx]


    def fit(self, X, y, save_weights=False):
        samples, features = X.shape
        class_labels = np.where(y <= 0, -1, 1)
        self.initialize_weights(n_features=features)
        self.update_weights(class_labels=class_labels, X=X)
        if save_weights:
            np.save(f"weights/weights_{self.learning_rate}_{self.lambda_param}.npy", self.w)
            np.save(f"weights/bias_{self.learning_rate}_{self.lambda_param}.npy", self.b)
        return self.w, self.b

    def predict(self, X):
        prediction = np.dot(X, self.w) - self.b
        return np.sign(prediction)
