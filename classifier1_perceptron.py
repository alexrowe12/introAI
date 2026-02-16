"""
Classifier 1: Manually Crafted Single Perceptron for X vs O Classification

Key Insight:
- X's have ink in the CENTER where the two lines cross
- O's have an EMPTY CENTER (they're hollow circles)

Strategy:
- Crop to center 40% of image (removes frame, keeps shape)
- Detect ink using contrast from whiteboard background
- The perceptron computes: center_intensity - ring_intensity
- X → positive (ink in center) → output 1
- O → negative (empty center, ink in ring) → output 0
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import argparse


class ManualPerceptron(nn.Module):
    """
    A single perceptron with manually selected weights.

    Computes: (ink in center) - (ink in ring)
    X's have crossing lines → more ink in center → POSITIVE
    O's are hollow circles → less ink in center → NEGATIVE
    """

    def __init__(self, grid_size=16):
        super().__init__()
        self.grid_size = grid_size

        weights = self._create_center_vs_ring_weights(grid_size)
        self.linear = nn.Linear(grid_size * grid_size, 1, bias=True)

        with torch.no_grad():
            self.linear.weight.data = torch.tensor(weights.flatten(), dtype=torch.float32).unsqueeze(0)
            self.linear.bias.data = torch.tensor([0.0], dtype=torch.float32)

    def _create_center_vs_ring_weights(self, size):
        """
        Weights that compute: (center intensity) - (ring intensity)
        - Center zone (dist < 0.35): POSITIVE
        - Ring zone (0.35 < dist < 0.7): NEGATIVE
        - Outer zone (> 0.7): ZERO (ignored)
        """
        weights = np.zeros((size, size), dtype=np.float32)
        center = size / 2

        center_count = 0
        ring_count = 0

        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center + 0.5)**2 + (j - center + 0.5)**2)
                max_dist = np.sqrt(2) * center
                normalized_dist = dist / max_dist

                if normalized_dist < 0.35:
                    weights[i, j] = 1.0
                    center_count += 1
                elif normalized_dist < 0.7:
                    weights[i, j] = -1.0
                    ring_count += 1

        if center_count > 0:
            weights[weights > 0] /= center_count
        if ring_count > 0:
            weights[weights < 0] /= ring_count

        return weights

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            return (self.forward(x) > 0).int().squeeze()

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x) * 15).squeeze()


def preprocess_image(image_path, grid_size=16, debug=False):
    """
    Preprocess image: crop center, detect ink, downsample.
    """
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    h, w = img_array.shape

    # Crop to center 40% - this should capture the shape and exclude most of the frame
    margin_h = int(h * 0.30)
    margin_w = int(w * 0.30)
    cropped = img_array[margin_h:h-margin_h, margin_w:w-margin_w]

    # Invert so ink = bright
    cropped = 255 - cropped

    # Adaptive thresholding: top percentile = ink
    threshold = np.percentile(cropped, 88)
    ink_mask = (cropped > threshold).astype(np.float32)

    # Resize to grid_size x grid_size
    ink_img = Image.fromarray((ink_mask * 255).astype(np.uint8))
    resized = ink_img.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    resized_array = np.array(resized, dtype=np.float32) / 255.0

    if debug:
        print(f"Image: {Path(image_path).name}")
        print(f"Cropped to center: [{margin_h}:{h-margin_h}, {margin_w}:{w-margin_w}]")
        print(f"Ink coverage: {resized_array.mean():.1%}")

        # 8x8 visualization
        display = np.array(Image.fromarray((resized_array * 255).astype(np.uint8)).resize(
            (8, 8), Image.Resampling.BILINEAR)) / 255.0
        print("Grid (# = ink):")
        for row in display:
            print(" ".join(['#' if v > 0.25 else '.' for v in row]))
        print()

    return torch.tensor(resized_array.flatten(), dtype=torch.float32)


def classify_image(image_path, model=None, grid_size=16, debug=False):
    """Classify a single image as X or O."""
    if model is None:
        model = ManualPerceptron(grid_size=grid_size)

    features = preprocess_image(image_path, grid_size=grid_size, debug=debug)
    features = features.unsqueeze(0)

    with torch.no_grad():
        raw_score = model.forward(features).item()

    pred = 1 if raw_score > 0 else 0
    prob = model.predict_proba(features).item()
    label = "X" if pred == 1 else "O"
    confidence = prob if pred == 1 else 1 - prob

    return label, confidence, raw_score


def test_on_dataset(image_dir, labels_dict=None, grid_size=16, debug=False):
    """Test the classifier on a directory of images."""
    model = ManualPerceptron(grid_size=grid_size)
    image_dir = Path(image_dir)

    results = []
    correct = 0
    total = 0

    # Support both .jpg and .jpeg extensions
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    for img_path in sorted(image_files):
        pred_label, confidence, raw_score = classify_image(img_path, model, grid_size, debug)

        result = {
            "file": img_path.name,
            "prediction": pred_label,
            "confidence": confidence,
            "raw_score": raw_score
        }

        if labels_dict and img_path.name in labels_dict:
            true_label = labels_dict[img_path.name]
            result["true_label"] = true_label
            result["correct"] = pred_label == true_label
            if result["correct"]:
                correct += 1
            total += 1

        results.append(result)
        status = ""
        if labels_dict and img_path.name in labels_dict:
            status = " ✓" if result["correct"] else f" ✗ (true: {true_label})"
        print(f"{img_path.name}: {pred_label} (score: {raw_score:+.4f}, conf: {confidence:.1%}){status}")

    if total > 0:
        accuracy = correct / total
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"(Random guessing = 50%, need >50% to pass)")
        return {"results": results, "accuracy": accuracy}

    return {"results": results}


def visualize_weights(grid_size=16):
    """Visualize the perceptron weights."""
    model = ManualPerceptron(grid_size=grid_size)
    weights = model.linear.weight.data.numpy().reshape(grid_size, grid_size)

    print("Weight zones: + = center (positive), - = ring (negative), . = outer")
    print()

    display = np.array(Image.fromarray(weights).resize((8, 8), Image.Resampling.BILINEAR))
    max_abs = np.abs(display).max()

    for row in display:
        line = ""
        for w in row:
            if w > 0.02 * max_abs:
                line += "+ "
            elif w < -0.02 * max_abs:
                line += "- "
            else:
                line += ". "
        print(line)

    print(f"\nCenter weights sum to: {weights[weights > 0].sum():.3f}")
    print(f"Ring weights sum to: {weights[weights < 0].sum():.3f}")


# Ground truth labels
LABELS = {
    # images/ directory
    "IMG_3134.jpg": "X",
    "IMG_3135.jpg": "O",
    "IMG_3136.jpg": "X",
    "IMG_3137.jpg": "O",
    "IMG_3138.jpg": "X",
    "IMG_3139.jpg": "O",
    "IMG_3140.jpg": "X",
    "IMG_3141.jpg": "O",
    "IMG_3142.jpg": "O",
    "IMG_3143.jpg": "X",
    # images2/ directory (sorted alphabetically: 1, 10, 2, 3, 4, 5, 6, 7, 8, 9)
    "image1.jpeg": "O",
    "image10.jpeg": "O",
    "image2.jpeg": "X",
    "image3.jpeg": "O",
    "image4.jpeg": "X",
    "image5.jpeg": "O",
    "image6.jpeg": "O",
    "image7.jpeg": "X",
    "image8.jpeg": "X",
    "image9.jpeg": "O",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify X vs O images using a manually crafted perceptron")
    parser.add_argument("-d", "--dir", type=str, default="images",
                        help="Directory containing images to classify (default: images)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed debug output for each image")
    args = parser.parse_args()

    print("=" * 60)
    print("Classifier 1: Manually Crafted Perceptron")
    print("=" * 60)
    print()

    print("CONCEPT:")
    print("-" * 40)
    print("X's have ink where lines CROSS in the CENTER")
    print("O's are HOLLOW - empty center, ink in a ring")
    print()
    print("Perceptron computes: (center ink) - (ring ink)")
    print("  X → more center ink → POSITIVE score")
    print("  O → less center ink → NEGATIVE score")
    print()

    print("WEIGHT VISUALIZATION:")
    print("-" * 40)
    visualize_weights(grid_size=16)
    print()

    print("TESTING ON DATASET:")
    print("-" * 40)

    # Resolve image directory path
    image_dir = Path(args.dir)
    if not image_dir.is_absolute():
        image_dir = Path(__file__).parent / args.dir

    if image_dir.exists():
        print(f"Using image directory: {image_dir}")
        print()
        test_on_dataset(image_dir, labels_dict=LABELS, grid_size=16, debug=args.verbose)
    else:
        print(f"Error: Directory not found: {image_dir}")

    print()
    print("=" * 60)
    print("USAGE:")
    print("  python classifier1_perceptron.py -d <image_directory>")
    print("  python classifier1_perceptron.py -d images2 -v")
    print("=" * 60)
