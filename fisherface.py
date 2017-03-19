"""Fisherface (PCA+LDA) feature extraction."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class FisherfaceExtractor:
    def __init__(self, pca_components=80, lda_components=None):
        self.pca_components = pca_components
        self.lda_components = lda_components
        self.pca = None
        self.lda = None

    def fit(self, X, y):
        n_classes = len(set(y))
        if self.lda_components is None:
            self.lda_components = n_classes - 1
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        X_pca = self.pca.fit_transform(X)
        self.lda = LinearDiscriminantAnalysis(n_components=self.lda_components)
        self.lda.fit(X_pca, y)
        print(f"Fisherface: PCA({self.pca_components}) -> LDA({self.lda_components})")
        return self

    def transform(self, X):
        return self.lda.transform(self.pca.transform(X))

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
