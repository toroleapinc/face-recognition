"""Try combining all three feature sets."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from load_data import load_orl
from eigenface import EigenfaceExtractor
from fisherface import FisherfaceExtractor
from gabor import GaborExtractor

def main():
    X, y = load_orl()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    eigen = EigenfaceExtractor(80)
    X_e_tr = eigen.fit_transform(X_train)
    X_e_te = eigen.transform(X_test)

    fisher = FisherfaceExtractor(80)
    X_f_tr = fisher.fit_transform(X_train, y_train)
    X_f_te = fisher.transform(X_test)

    gabor = GaborExtractor()
    X_g_tr = gabor.fit_transform(X_train)
    X_g_te = gabor.transform(X_test)

    scaler_e = StandardScaler().fit(X_e_tr)
    scaler_f = StandardScaler().fit(X_f_tr)
    scaler_g = StandardScaler().fit(X_g_tr)

    X_combined_train = np.hstack([scaler_e.transform(X_e_tr), scaler_f.transform(X_f_tr), scaler_g.transform(X_g_tr)])
    X_combined_test = np.hstack([scaler_e.transform(X_e_te), scaler_f.transform(X_f_te), scaler_g.transform(X_g_te)])

    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_combined_train, y_train)
    acc = svm.score(X_combined_test, y_test)
    print(f"Combined features accuracy: {acc:.4f}")
    # TODO: add cross-validation for this

if __name__ == '__main__':
    main()
