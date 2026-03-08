import os
import numpy as np
from PIL import Image
np.random.seed(0)

IMG_SIZE = 64
LR = 0.01
EPOCHS = 1000
LAMBDA = 1e-3

def load_data():
    X, Y = [], []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    classes = {"not_trefoil": 0, "trefoil": 1}

    for cls, label in classes.items():
        folder = os.path.join(data_dir, cls)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            img = Image.open(img_path).convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img, dtype=np.float32) / 255.0

            X.append(img.flatten())
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)

    return X, Y

X, Y = load_data()

W1 = np.random.randn(4096, 256) * np.sqrt(2 / 4096)
b1 = np.zeros((1, 256))

W2 = np.random.randn(256, 128) * np.sqrt(2 / 256)
b2 = np.zeros((1, 128))

W3 = np.random.randn(128, 64) * np.sqrt(2 / 128)
b3 = np.zeros((1, 64))

W4 = np.random.randn(64, 32) * np.sqrt(2 / 64)
b4 = np.zeros((1, 32))

W5 = np.random.randn(32, 1) * np.sqrt(2 / 32)
b5 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(z):
    return (z > 0).astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    a3 = relu(z3)

    z4 = a3 @ W4 + b4
    a4 = relu(z4)

    z5 = a4 @ W5 + b5
    y_hat = sigmoid(z5)

    cache = (X, z1, a1, z2, a2, z3, a3, z4, a4, z5, y_hat)
    return y_hat, cache

def loss(y, y_hat):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    bce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    l2 = (
        np.sum(W1**2) + np.sum(W2**2) +
        np.sum(W3**2) + np.sum(W4**2) +
        np.sum(W5**2)
    )

    return bce + (LAMBDA / (2 * y.shape[0])) * l2

def backward(cache, Y, lr=0.001):
    global W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

    (X, z1, a1, z2, a2, z3, a3, z4, a4, z5, y_hat) = cache
    m = X.shape[0]

    dz5 = y_hat - Y
    dW5 = (a4.T @ dz5) / m + (LAMBDA / m) * W5
    db5 = dz5.mean(axis=0, keepdims=True)

    da4 = dz5 @ W5.T
    dz4 = da4 * relu_deriv(z4)
    dW4 = (a3.T @ dz4) / m + (LAMBDA / m) * W4
    db4 = dz4.mean(axis=0, keepdims=True)

    da3 = dz4 @ W4.T
    dz3 = da3 * relu_deriv(z3)
    dW3 = (a2.T @ dz3) / m + (LAMBDA / m) * W3
    db3 = dz3.mean(axis=0, keepdims=True)

    da2 = dz3 @ W3.T
    dz2 = da2 * relu_deriv(z2)
    dW2 = (a1.T @ dz2) / m + (LAMBDA / m) * W2
    db2 = dz2.mean(axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = (X.T  @ dz1) / m + (LAMBDA / m) * W1
    db1 = dz1.mean(axis=0, keepdims=True)

    W5 -= lr * dW5
    b5 -= lr * db5
    W4 -= lr * dW4
    b4 -= lr * db4
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


for epoch in range(EPOCHS):
    y_hat, cache = forward(X)
    l = loss(Y, y_hat)
    backward(cache, Y, LR)

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {l:.4f}")

def predict_image(img_path, threshold=0.5):
    img = Image.open(img_path).convert("L")
    img = img.resize((64, 64))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.flatten().reshape(1, -1)

    y_hat, _ = forward(img)
    prob = y_hat[0, 0]

    label = "Trefoil" if prob >= threshold else "Not Trefoil"
    return prob, label

print("\nExample prediction:")
print("Trefoil prob:", predict_image(os.path.join("data", "trefoil", os.listdir("data/trefoil")[0])))
print("\nTest prediction:")
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
test_img_path = os.path.join(data_dir, "test", "img.png")
prob, label = predict_image(test_img_path)

print(f"Prediction: {label}")
print(f"Probability: {prob:.4f}")