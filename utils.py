"""Misc utilities."""
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def combine_features(*feature_sets_train, test_sets=None):
    combined_train = np.hstack(feature_sets_train)
    if test_sets:
        combined_test = np.hstack(test_sets)
        return combined_train, combined_test
    return combined_train
