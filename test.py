import enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools

class svc:
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        

    def predict(self, x_test):
        y_pred = []
        for xi in x_test:
            y_pred.append(np.sign(np.dot(self.w, xi) + self.b))
        return y_pred

if __name__ == "__main__":
    # Import data
    spotifyData = pd.read_csv('data1.csv')
    
    # Drop irrelevant classes and seperate features and classes
    x = spotifyData[['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
    y = spotifyData[['target']]

    # Convert milliseconds to seconds
    x['duration_ms'] = x['duration_ms']/1000
    x.rename(columns= {"duration_ms": "duration_s"}, inplace= True)

    # Min Max Normalization to create range [0,1]
    for feature in x:
        maxF = max(x[feature])
        minF = min(x[feature])
        x[feature] = x[feature] - minF
        x[feature] = x[feature]/(maxF - minF)

    # Z-score normalization
    for feature in x:
        meanF = x[feature].mean()
        stdF = x[feature].std()
        x[feature] = x[feature] - meanF
        x[feature] = x[feature]/stdF

    # Convert from pd.DataFrame to np.array
    x = x.to_numpy()
    y = y.to_numpy()

    # Create covariance matrix
    x_T = x.T
    cov_matrix = np.cov(x_T)

    # Eigen decomposition
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    eig = {}
    for i in range(len(eig_values)):
        eig[eig_values[i]] = eig_vectors[i]

    # Create projections
    projection_1 = x.dot(eig_vectors.T[0])
    projection_2 = x.dot(eig_vectors.T[1])

    # Create numpy array of projections
    pca = pd.DataFrame(projection_1, columns=['P1'])
    pca['P2'] = projection_2
    x = pca.to_numpy()

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.99)

    # Create and fit model
    svc = svc()
    model = svc.fit(x_train, y_train)

    # # Predict using model
    # y_pred = model.predict(x_test)

    # # Print Metrics
    # print("\nAccuracy Score: ")
    # print(accuracy_score(y_test, y_pred))