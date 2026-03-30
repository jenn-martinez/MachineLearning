import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.weights is None or len(self.weights) != n_features:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errors = 0
            for idx, xi in enumerate(X):
                linear_output = np.dot(xi, self.weights) + self.bias

                y_pred = 1 if linear_output >= 0 else 0
                update = self.lr * (y[idx] - y_pred)

                if update !=0.0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1

            self.errors_per_epoch.append(errors)
            if errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
    
    def activation_func(self, x):
        return np.where(x >= 0, 1, 0)