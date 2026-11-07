"""NumPy MLP for binary trefoil image classification.

This module preserves the original five-layer fully connected architecture and
full-batch gradient-descent training, but avoids training as an import side effect.
"""

from pathlib import Path
import numpy as np
from PIL import Image

IMG_SIZE = 64
DEFAULT_LR = 0.01
DEFAULT_EPOCHS = 1000
DEFAULT_L2 = 1e-3


def load_data(data_dir, image_size=IMG_SIZE):
    data_dir = Path(data_dir)
    images, labels = [], []
    for class_name, label in (("not_trefoil", 0), ("trefoil", 1)):
        folder = data_dir / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class directory: {folder}")
        for image_path in sorted(p for p in folder.iterdir() if p.is_file()):
            try:
                with Image.open(image_path) as image:
                    image = image.convert("L").resize((image_size, image_size))
                    images.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
                    labels.append(label)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not read image {image_path}: {exc}") from exc
    if not images:
        raise ValueError(f"No images found in {data_dir}")
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1)


class TrefoilMLP:
    """Five-layer dense binary classifier with ReLU hidden units and sigmoid output."""

    def __init__(self, input_size=IMG_SIZE * IMG_SIZE, seed=0, l2=DEFAULT_L2):
        rng = np.random.default_rng(seed)
        sizes = [input_size, 256, 128, 64, 32, 1]
        self.weights = [rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((1, b), dtype=np.float32) for b in sizes[1:]]
        self.l2 = l2

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        activations = [x]
        preactivations = []
        current = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ weight + bias
            preactivations.append(z)
            current = np.maximum(0, z)
            activations.append(current)
        z = current @ self.weights[-1] + self.biases[-1]
        preactivations.append(z)
        output = self._sigmoid(z)
        activations.append(output)
        return output, (activations, preactivations)

    def loss(self, y, prediction):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        bce = -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
        penalty = sum(np.sum(weight ** 2) for weight in self.weights)
        return float(bce + self.l2 * penalty / (2 * y.shape[0]))

    def train(self, x, y, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LR, log_every=200):
        if x.ndim != 2 or x.shape[1] != self.weights[0].shape[0]:
            raise ValueError("Input must be a 2D array with the configured flattened image size.")
        history = []
        count = x.shape[0]
        for epoch in range(epochs):
            prediction, (activations, preactivations) = self.forward(x)
            current_loss = self.loss(y, prediction)
            delta = prediction - y
            grad_w = [None] * len(self.weights)
            grad_b = [None] * len(self.biases)
            for layer in reversed(range(len(self.weights))):
                grad_w[layer] = activations[layer].T @ delta / count + (self.l2 / count) * self.weights[layer]
                grad_b[layer] = delta.mean(axis=0, keepdims=True)
                if layer > 0:
                    delta = (delta @ self.weights[layer].T) * (preactivations[layer - 1] > 0)
            for layer in range(len(self.weights)):
                self.weights[layer] -= learning_rate * grad_w[layer]
                self.biases[layer] -= learning_rate * grad_b[layer]
            history.append(current_loss)
            if log_every and epoch % log_every == 0:
                print(f"Epoch {epoch} | Loss: {current_loss:.4f}")
        return history

    def predict(self, x, threshold=0.5):
        probabilities, _ = self.forward(x)
        return probabilities, (probabilities >= threshold).astype(np.int64)


def load_image(path, image_size=IMG_SIZE):
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32).reshape(1, -1) / 255.0


def train_from_directory(data_dir, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LR, seed=0):
    x, y = load_data(data_dir)
    model = TrefoilMLP(seed=seed)
    model.train(x, y, epochs=epochs, learning_rate=learning_rate)
    return model
