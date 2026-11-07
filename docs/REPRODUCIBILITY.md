# Reproducibility and implementation notes

## Environment

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The trefoil classifier uses NumPy and Pillow. The crossing-number script imports OpenCV, scikit-image, NetworkX, NumPy, and Matplotlib. The dependency list includes both sets.

## Trefoil classifier

Run from the classifier directory:

```bash
cd "Deep Neural Network"
python NN.py
```

The script expects `data/not_trefoil/`, `data/trefoil/`, and `data/test/img.png` relative to that directory. It resizes grayscale images to 64×64, flattens them, trains a fully connected network, and prints an example prediction.

The script sets NumPy's seed to zero. It does not currently define a train/validation split or save model weights. The test folder is used for a single example prediction, not a reported held-out evaluation protocol.

## Crossing-number detector

The intended pipeline is grayscale loading → Otsu thresholding → skeletonization → 8-neighbour graph construction → candidate crossing detection by node degree and spatial filtering.

Known issues in the checked-in original script:
- It calls `os.path` but does not import `os`.
- It constructs a path named `image`, while the checked-in asset is named `image.png`.
- The script's interactive loop executes at module level.

These are documented rather than silently changed as part of this structural cleanup. Fixing them should be a separate, reviewable code-change commit.

## Reproducibility

Record Python and package versions, the exact input image, image preprocessing parameters, network hyperparameters, and random seeds. The classifier currently hard-codes image size, learning rate, epoch count, and regularization strength. The crossing detector uses a manually adjusted neighbourhood radius.

## Limitations

The available class images are small in number. Results from this dataset should be treated as a proof of concept, not evidence of generalization to arbitrary knot diagrams.
