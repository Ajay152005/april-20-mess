import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures




class KNN:
    def __init__(self, k=3, distance_metric = 'euclidean', weights='uniform', missing_strategy = 'mean'):
        self.weights = weights
        self.missing_strategy = missing_strategy
        self.distance_metric = distance_metric
        self.k = k
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _get_distance(self, x1, x2):
        if self.distance_metric =='euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'manhattan'.")
    
    def _handle_missing_values(self, X):
        if self.missing_strategy == 'mean':
            column_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(column_means, inds[1])
            return X
        else:
            raise ValueError("Unsupported missing strategy. Choose 'mean'.")
    
    def _get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones_like(distances)
        
        elif self.weights == 'distance':
            return 1 / (distances + 1e-8) #Adding a small value to avoid division by zero
        else:
            raise ValueError("Unsupported weighting strategy. Choose 'uniform' or 'distance'")

    def fit(self, X_train , y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def _predict_single_classification(self, sample):
        distances = [self._get_distance(sample, x) for x in self.X_train]
        nearest_neighbors = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_neighbors]
        if self.weights == 'uniform':
            majority_vote = Counter(nearest_labels).most_common(1)
        else:
            weights = self._get_weights(np.array(distances)[nearest_neighbors])
            label_counts = Counter(nearest_labels, weights= weights)
            majority_vote = label_counts.most_common(1)
        return majority_vote[0][0]
    
    def _predict_single_regression(self, sample):
        distances = [self._get_distance(sample, x) for x in self.X_train]
        nearest_neighbors = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_neighbors]
        if self.weights == 'uniform':
            return np.mean(nearest_labels)
        else:
            weights = self._get_weights(np.array(distances)[nearest_neighbors])
            return np.sum(weights * nearest_labels) / np.sum(weights)
        
    def predict(self, X_test):
        if len(X_test.shape) == 1:
            X_test = np.array([X_test])
        if isinstance(self.y_train[0], (int, float)):
            return [self._predict_single_regression(sample) for sample in X_test]
        else:
            return [self._predict_single_classification(sample) for sample in X_test]
        
#load iris dataset

iris = load_iris()
X, y = iris.data, iris.target
# adding two random features to the dataset
random_features = np.random.rand(X.shape[0], 2)
X = np.hstack((X, random_features))

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Compute staistical features

statistical_features = np.hstack((np.mean(X, axis=1, keepdims= True),
                                  np.var(X, axis=1, keepdims=True),
                                  np.nanmedian(X, axis=1, keepdims= True),
                                  np.nanstd(X, axis=1, keepdims=True),
                                  np.nanpercentile(X, 25, axis=1, keepdims=True),
                                  np.anpercentile(X, 75, axis=1, keepdims=True),
                                  np.nanpercentile(X, 10, axis=1, keepdims = True),
                                  np.nanpercentile(X, 90, axis=1, keepdims=True),
                                  np.nanpercentile(X, 5, axis= 1, keepdims = True),
                                  np.nanpercentile(X, 95, axis=1, keepdims= True),
                                  np.nanpercentile(X, 1, axis=1, keepdims=True),
                                  np.nanpercentile(X, 99, axis=1,keepdims=True),
                                  np.nanmax(X, axis=1, keepdims= True),
                                  np.nanmin(X, axis=1, keepdims=True),
                                  np.nanmean(np.diff(X, axis=1), axis=1, keepdims=True),
                                  np.nansum(X, axis=1, keepdims=True)))

#Combine original features, polynomial features, and statistical features
X_combined = np.hstack((X, X_poly, statistical_features))

# Create interaction terms
interaction_features = np.multiply(X[:, 0], X[:,1])[:, np.newaxis]

# Example: Adding BMI as a feature
height = X[:, 0]
weight = X[: , 1]
bmi = weight / (height ** 2)
bmi = bmi[:, np.newaxis]

# Example : Creating a feature representing the ratio of two features
ratio_feature = X[: ,2] / X[:,3]
ratio_feature = ratio_feature[:, np.newaxis]

# Example: Using CountVectorizer to convert text data into features
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(text_data)

# Example: Adding lag features
lag_1 = np.roll(X[:, 0], 1)

lag_1[0] = np.nan # set the first value to NaN
lag_1 = lag_1[:, np.newaxis]


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create KNN model

model = KNN(k=3, weights = 'uniform', missing_strategy='mean')
model.fit(X_train,y_train)

#Make predictions
predictions = model.predict(X_test)

#visualize the results

def plot_results(X, y, predictions):
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', 'D']

    # Plot actual data points
    for i in range(len(np.unique(y))):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=colors[i], marker=markers[i], label=f'Class {i}')

    # Plot predicted data points
    for i in range(len(np.unique(predictions))):
        plt.scatter(X_test[np.array(predictions) == i, 0], X_test[np.array(predictions) == i, 1], color=colors[i], marker='x', label=f'Predicted Class {i}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Classification Results')
    plt.legend()
    plt.show()



plot_results(X_test[: ,:2], y_test, predictions)