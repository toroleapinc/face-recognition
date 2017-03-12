"""Load ORL face database."""
import os
import cv2
import numpy as np

def load_orl(root="data/orl_faces", size=(112, 92)):
    images = []
    labels = []
    for subject_dir in sorted(os.listdir(root)):
        subject_path = os.path.join(root, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        label = int(subject_dir.replace("s", "")) - 1
        for img_file in sorted(os.listdir(subject_path)):
            img = cv2.imread(os.path.join(subject_path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (size[1], size[0]))
                images.append(img.flatten())
                labels.append(label)
    return np.array(images, dtype=np.float64), np.array(labels)

if __name__ == "__main__":
    X, y = load_orl()
    print(f"Loaded {X.shape[0]} images, {len(set(y))} subjects")
    print(f"Image vector size: {X.shape[1]}")
