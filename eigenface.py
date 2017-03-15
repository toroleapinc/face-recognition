"""Eigenface (PCA) feature extraction."""
import numpy as np
from sklearn.decomposition import PCA

class EigenfaceExtractor:
    def __init__(self, n_components=80):
        self.n_components = n_components
        self.pca = None
        self.mean_face = None

    def fit(self, X):
        self.mean_face = np.mean(X, axis=0)
        self.pca = PCA(n_components=self.n_components, whiten=True)
        self.pca.fit(X)
        explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA: {self.n_components} components, {explained:.1f}% variance")
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def eigenfaces(self):
        return self.pca.components_
