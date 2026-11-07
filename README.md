# Deep neural networks for knot theory

An exploratory project applying image processing and a small neural network to knot diagrams. It contains two separate experiments:

1. **Trefoil classifier** — a NumPy fully connected neural network classifying images as trefoil or not-trefoil.
2. **Crossing-candidate detector** — an image-processing pipeline that thresholds a knot image, skeletonizes it, builds a pixel-neighbour graph, and counts candidate crossings. This is not a neural network and is only a heuristic image-based estimate.

The project was carried out under Prof. Rama Mishra at IISER Pune as a summer-student project.

## Modular implementation (recommended)

The new package is under `src/knot_theory/`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Train the classifier and predict the bundled test image:

```bash
python -m knot_theory.cli classify
```

Estimate crossing candidates in the bundled diagram:

```bash
python -m knot_theory.cli crossings
```

Use `--help` after either subcommand for options. Run commands from the repository root.

## Project structure

- `src/knot_theory/classifier.py` — data loading, MLP, training, and prediction.
- `src/knot_theory/crossings.py` — thresholding, skeletonization, graph construction, and candidate detection.
- `src/knot_theory/cli.py` — command-line interface.
- `tests/` — small unit tests for the classifier and graph pipeline.
- `Deep Neural Network/NN.py` — original classifier script, retained unchanged.
- `Deep Neural Network/data/` — class images and test image.
- `Crossing Number Predictor/Code.py` — original crossing detector, retained unchanged.
- `Crossing Number Predictor/image.png` — bundled sample diagram.

## Method overview

### Trefoil classifier
Images are converted to grayscale, resized to 64×64, normalized to [0, 1], and flattened. A five-layer dense network uses ReLU hidden layers and a sigmoid output, trained with binary cross-entropy plus L2 regularization.

### Crossing candidates
The detector applies inverted Otsu thresholding, skeletonization, 8-neighbour graph construction, and degree-based candidate detection with spatial de-duplication. The output is a candidate count, not a mathematical proof of the knot's crossing number.

## Original code and reproducibility

Both original scripts and all image data remain in their original directories. The modular implementation avoids running training or interactive prompts merely by importing modules. It is a cleaned port, not guaranteed to reproduce the exact legacy numerical output; see [reproducibility notes](docs/REPRODUCIBILITY.md).

## Tests

```bash
python -m unittest discover -s tests
```

These tests check basic model operations and crossing-graph behavior. They do not establish classification accuracy or mathematical correctness for arbitrary knot diagrams.
