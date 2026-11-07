# Reproducibility and implementation notes

## Install and run

Install the root-level dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m knot_theory.cli classify --help
python -m knot_theory.cli crossings --help
```

Run from the repository root so the default paths resolve to the checked-in image folders.

## Trefoil classifier

The default classifier command reads images from `Deep Neural Network/data/not_trefoil/` and `Deep Neural Network/data/trefoil/`, trains the five-layer dense network for 1,000 full-batch epochs, then predicts `Deep Neural Network/data/test/img.png`.

Images are grayscale, resized to 64×64, and normalized to [0, 1]. The network uses ReLU hidden layers, sigmoid output, binary cross-entropy, and L2 weight regularization. Defaults are seed 0, learning rate 0.01, and L2 coefficient 1e-3. The split is not divided into training/validation sets; the bundled test image is one example, not a robust test protocol.

The modular implementation uses NumPy's `default_rng` for initialization and float32 arrays. This differs from the legacy script's global `np.random.seed(0)` initialization, so bit-for-bit or metric equivalence is not promised. The original `NN.py` remains untouched.

## Crossing-candidate detector

The default image is `Crossing Number Predictor/image.png`. The pipeline is grayscale loading → inverted Otsu thresholding → skeletonization → 8-neighbour graph → degree-at-least-four candidate nodes → radius-based de-duplication.

The method is heuristic. Skeleton pixel connectivity can create clusters or spurious high-degree nodes, and the result should not be presented as a mathematically certified crossing number. The modular CLI fixes the legacy script's missing `os` import and incorrect default filename; the original script is preserved unchanged.

## Tests and limitations

Run `python -m unittest discover -s tests`. Tests cover model forward/prediction shape, a small parameter update, and a synthetic crossing-shaped skeleton. They do not train on the full image collection or validate crossing detection on a benchmark dataset.

For research reporting, record Python, NumPy, Pillow, OpenCV, scikit-image, NetworkX versions; random seed; image preprocessing; model hyperparameters; and the exact input images.
