"""Evaluate trained models."""
import argparse
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from load_data import load_orl
from utils import normalize_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/orl_faces')
    parser.add_argument('--model-file', default='models.pkl')
    parser.add_argument('--test-size', type=float, default=0.3)
    args = parser.parse_args()

    X, y = load_orl(args.data)
    _, X_test, _, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    with open(args.model_file, 'rb') as f:
        models = pickle.load(f)

    for name, (extractor, svm) in models.items():
        print(f"\n=== {name.upper()} ===")
        X_test_feat = extractor.transform(X_test)
        # need to normalize but we don't have train features here...
        # this is a bit hacky, should save the scaler too
        y_pred = svm.predict(X_test_feat)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == '__main__':
    main()
