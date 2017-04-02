"""Train SVM classifiers on different feature sets."""
import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from load_data import load_orl
from eigenface import EigenfaceExtractor
from fisherface import FisherfaceExtractor
from gabor import GaborExtractor
from utils import normalize_features

def train_svm(X_train, y_train, kernel='rbf'):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001]}
    if kernel == 'linear':
        param_grid = {'C': [0.1, 1, 10, 100]}
    grid = GridSearchCV(SVC(kernel=kernel), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}, CV acc: {grid.best_score_:.4f}")
    return grid.best_estimator_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/orl_faces')
    parser.add_argument('--pca-components', type=int, default=80)
    parser.add_argument('--test-size', type=float, default=0.3)
    args = parser.parse_args()

    X, y = load_orl(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42  # same split as evaluate
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("\n--- Eigenface ---")
    eigen = EigenfaceExtractor(n_components=args.pca_components)
    X_eigen_train = eigen.fit_transform(X_train)
    X_eigen_test = eigen.transform(X_test)
    X_eigen_train, X_eigen_test = normalize_features(X_eigen_train, X_eigen_test)
    svm_eigen = train_svm(X_eigen_train, y_train)

    print("\n--- Fisherface ---")
    fisher = FisherfaceExtractor(pca_components=args.pca_components)
    X_fisher_train = fisher.fit_transform(X_train, y_train)
    X_fisher_test = fisher.transform(X_test)
    X_fisher_train, X_fisher_test = normalize_features(X_fisher_train, X_fisher_test)
    svm_fisher = train_svm(X_fisher_train, y_train)

    print("\n--- Gabor ---")
    gabor_ext = GaborExtractor()
    X_gabor_train = gabor_ext.fit_transform(X_train)
    X_gabor_test = gabor_ext.transform(X_test)
    X_gabor_train, X_gabor_test = normalize_features(X_gabor_train, X_gabor_test)
    svm_gabor = train_svm(X_gabor_train, y_train)

    models = {
        'eigenface': (eigen, svm_eigen),
        'fisherface': (fisher, svm_fisher),
        'gabor': (gabor_ext, svm_gabor),
    }
    with open('models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("\nModels saved to models.pkl")

if __name__ == '__main__':
    main()
