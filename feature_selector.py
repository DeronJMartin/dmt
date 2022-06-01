from email.policy import default
import numpy as np
import pandas as pd

class feature_selector:
    def __init__(self):
        pass

    def select(self, x_train, y_train, x_test, y_test, classifier):
        
        n_rows, n_cols = x_train.shape

        all_features = set(range(n_cols))
        features = []
        accuracy = []
        # current_feature = 

        # while