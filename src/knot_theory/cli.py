"""Command-line interface for the classifier and crossing detector."""

import argparse
from pathlib import Path
from .classifier import train_from_directory, load_image
from .crossings import estimate_crossing_number


def main():
    parser = argparse.ArgumentParser(description="Knot-image experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    classifier = subparsers.add_parser("classify", help="Train the trefoil/not-trefoil MLP and predict an image")
    classifier.add_argument("--data-dir", default="Deep Neural Network/data")
    classifier.add_argument("--image", default="Deep Neural Network/data/test/img.png")
    classifier.add_argument("--epochs", type=int, default=1000)
    classifier.add_argument("--learning-rate", type=float, default=0.01)
    classifier.add_argument("--seed", type=int, default=0)
    crossings = subparsers.add_parser("crossings", help="Estimate crossing candidates in a knot diagram")
    crossings.add_argument("--image", default="Crossing Number Predictor/image.png")
    crossings.add_argument("--radius", type=float, default=5)
    args = parser.parse_args()

    if args.command == "classify":
        model = train_from_directory(args.data_dir, args.epochs, args.learning_rate, args.seed)
        probability, label = model.predict(load_image(args.image))
        print(f"Trefoil probability: {float(probability[0, 0]):.4f}")
        print("Prediction:", "Trefoil" if int(label[0, 0]) else "Not Trefoil")
    else:
        count, _, _ = estimate_crossing_number(args.image, args.radius)
        print(f"Estimated crossing candidates: {count}")
        print("Note: this is a heuristic image-based estimate, not a mathematical knot invariant.")


if __name__ == "__main__":
    main()
