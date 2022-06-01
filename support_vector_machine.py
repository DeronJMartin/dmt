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

        # Convert class label 0 into -1 for easier calculations
        for i in range(len(self.y_train)):
            if self.y_train[i] == 0:
                self.y_train[i] = -1
        
        # Create dictionary to store Support Vectors
        vectors = {}

        # Create transforms for w
        transforms = [[1,1], [1,-1], [-1,1], [-1,-1]]

        # Create list of all values
        feature_values = []
        for row in self.x_train:
            for value in row:
                feature_values.append(value)

        # Save min and max values
        self.max_feature_value = max(feature_values)
        self.min_feature_value = min(feature_values)
        
        # Define step sizes for calculating w
        w_step_sizes = [self.max_feature_value*0.1, self.max_feature_value*0.01, self.max_feature_value * 0.001]

        # Define step sizes for calculating b
        b_range_multiplier = 2
        b_multiplier = 5
        
        # Vector multiplier
        vec_mult = 2

        # Optimization
        current_optimum = self.max_feature_value * vec_mult

        for step in w_step_sizes:
            w = np.array([current_optimum , current_optimum])
            optimized = False
            while optimized != True:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiplier), self.max_feature_value * b_range_multiplier, step * b_multiplier):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        for i in range(len(self.x_train)):
                            yi = self.y_train[i]
                            xi = self.x_train[i]
                            print(w_t, b, yi * (np.dot(w_t, xi) + b))
                            if (yi * (np.dot(w_t, xi) + b)) < 1:
                                found_option = False
                                break
                        if found_option:
                            vectors[np.linalg.norm(w_t)] = [w_t,b]
                            vecs = pd.DataFrame(vectors)
                            vecs.to_csv('vectors.csv', index= None, header= False)
            
                if w[0] <= self.min_feature_value:
                    optimized = True
                    print("Optimized step")
                else:
                    w = w - step
        
            # Save support vector
            norms = sorted([v for v in vectors])
            vector = vectors[norms[0]]
            self.w = vector[0]
            self.b = vector[1]
            current_optimum = vector[0] + step * vec_mult

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    # Create and fit model
    svc = svc()
    model = svc.fit(x_train, y_train)

    # Predict using model
    y_pred = model.predict(x_test)

    # Print Metrics
    print("\nAccuracy Score: ")
    print(accuracy_score(y_test, y_pred))