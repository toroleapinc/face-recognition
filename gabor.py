"""Gabor filter feature extraction."""
import numpy as np
from skimage.filters import gabor
from sklearn.decomposition import PCA

class GaborExtractor:
    def __init__(self, frequencies=(0.1, 0.2, 0.3, 0.4), orientations=8, pca_components=100):
        self.frequencies = frequencies
        self.n_orientations = orientations
        self.pca_components = pca_components
        self.pca = None

    def _extract_gabor(self, image, shape):
        img = image.reshape(shape)
        features = []
        for freq in self.frequencies:
            for i in range(self.n_orientations):
                theta = i * np.pi / self.n_orientations
                filt_real, filt_imag = gabor(img, frequency=freq, theta=theta)
                features.append(filt_real.mean())
                features.append(filt_real.var())
                features.append(filt_imag.mean())
                features.append(filt_imag.var())
        return np.array(features)

    def fit_transform(self, X, image_shape=(112, 92)):
        print("Extracting Gabor features...")
        gabor_feats = np.array([self._extract_gabor(x, image_shape) for x in X])
        if self.pca_components and gabor_feats.shape[1] > self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            gabor_feats = self.pca.fit_transform(gabor_feats)
        print(f"Gabor features shape: {gabor_feats.shape}")
        return gabor_feats

    def transform(self, X, image_shape=(112, 92)):
        gabor_feats = np.array([self._extract_gabor(x, image_shape) for x in X])
        if self.pca is not None:
            gabor_feats = self.pca.transform(gabor_feats)
        return gabor_feats
