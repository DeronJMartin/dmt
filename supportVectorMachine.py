import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SVC():
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        # Convert class label 0 into -1 for easier calculations
        for i in range(len(self.y_train)):
            if self.y_train[i] == 0:
                self.y_train[i] = -1
        
        # Create dictionary to store Support Vectors
        supportVectors = {}

        # Create and calculate max and min feature values
        self.max_feature_vector = x_train[0]
        self.min_feature_vector = x_train[0]

        for row in x_train:
            for i in range(len(row)):
                self.max_feature_vector[i] = max(self.max_feature_vector[i], row[i])
                self.min_feature_vector[i] = min(self.min_feature_vector[i], row[i])

        # Define step sizes for optimization
        step_sizes = [self.max_feature_vector*0.1, self.max_feature_vector*0.01, self.max_feature_vector*0.001]



    def predict(self, x_test):
        pass

if __name__ == "__main__":
    # Import data
    spotifyData = pd.read_csv('data1.csv')
    
    # Drop irrelevant classes and seperate features and classes
    x = spotifyData[['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
    y = spotifyData[['target']]

    # Convert from pd.DataFrame to np.array
    x = x.to_numpy()
    y = y.to_numpy()

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create and fit model
    svc = SVC()
    model = svc.fit(x_train, y_train)

    # Predict using model
    y_pred = model.predict(x_test)

    print(len(x_train))