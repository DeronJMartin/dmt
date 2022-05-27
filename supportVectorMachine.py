import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class supportVectorClassifier():
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test, y_test):
        pass

if __name__ == "__main__":
    # Import data
    spotifyData = pd.read_csv('data1.csv')
    
    # Drop irrelevant classes and seperate features and classes
    x = spotifyData.drop(['artist', 'target', 'song_title', 'serial_num'], axis=1)
    y = spotifyData['target']

    print(x.shape)