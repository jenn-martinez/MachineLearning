import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errors = 0
            for idx, xi in enumerate(X):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors_per_epoch.append(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)
    
    def activation_func(self, x):
        return np.where(x >= 0, 1, 0)