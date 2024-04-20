import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent

        for _ in range(self.n_iterations):
            #Compute prediction

            y_predicted = np.dot(X, self.weights) + self.bias

            # compute gradients

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted -y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        def predict(self, X):
            return np.dot(X, self.weights) + self.bias
        
# Example Usage
X_train = np.array([1], [2], [3],[4], [5])
y_train = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X_train, y_train)

X_test = np.array([[6], [7]])
predictions = model.predict(X_test)
print(predictions)