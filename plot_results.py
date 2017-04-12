"""Quick plots for eigenfaces and confusion matrices."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_eigenfaces(components, image_shape=(112, 92), n=16, save='eigenfaces.png'):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < min(n, len(components)):
            ax.imshow(components[i].reshape(image_shape), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save)
    print(f"Saved {save}")

def plot_confusion(cm, save='confusion.png'):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(save)

if __name__ == '__main__':
    cm = np.random.randint(0, 10, (40, 40))
    plot_confusion(cm)
