# Deep neural networks for knot theory

An exploratory project applying image processing and a small neural network to knot images. It contains two separate experiments:

1. **Trefoil classifier** — a NumPy-based fully connected neural network that classifies images as trefoil or non-trefoil.
2. **Crossing-number predictor** — an image-processing pipeline that thresholds a knot image, skeletonizes it, builds a pixel-neighbour graph, and counts candidate crossings. This second program is not a neural network.

The project was carried out under Prof. Rama Mishra at IISER Pune as a summer-student project.

## Repository layout

- `Deep Neural Network/NN.py` — original binary classifier.
- `Deep Neural Network/data/` — image examples grouped by class, plus a test image.
- `Crossing Number Predictor/Code.py` — original image-processing crossing detector.
- `Crossing Number Predictor/image.png` — example image asset.
- `docs/REPRODUCIBILITY.md` — setup and implementation notes.
- `requirements.txt` — dependencies.

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the trefoil classifier

```bash
cd "Deep Neural Network"
python NN.py
```

The script loads images from its local `data/` directory, trains the network, and evaluates the bundled test image. It currently trains as soon as the script is executed.

## Crossing-number detector

The crossing detector is a separate classical image-processing experiment, not an ML model. Its current script has path/import assumptions that need attention before it can be run reliably from a fresh clone; see [known issues](docs/REPRODUCIBILITY.md). The original code and image asset are preserved.

## Scope and limitations

These are educational prototypes with small image datasets and simple implementations. They should not be treated as general knot invariants or robust classifiers without additional validation. No code or data has been removed in this cleanup branch.
