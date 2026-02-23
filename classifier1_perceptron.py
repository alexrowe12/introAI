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
from PIL import Image, ImageOps
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
        - Center zone (dist < 0.28): POSITIVE
        - Ring zone (0.28 < dist < 0.65): NEGATIVE
        - Outer zone (> 0.65): ZERO (ignored)
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

                if normalized_dist < 0.28:
                    weights[i, j] = 1.0
                    center_count += 1
                elif normalized_dist < 0.65:
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
    Preprocess image:
    1. Correct EXIF orientation
    2. Convert to grayscale
    3. Otsu threshold to find ink
    4. Find bounding box of all ink (the drawn frame + shape)
    5. Shrink inward by 15% on each side to strip the frame border
    6. Resize remaining region to grid_size x grid_size
    """
    img = ImageOps.exif_transpose(Image.open(image_path)).convert('L')
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape

    # Otsu threshold: find the natural split between ink and background
    flat = img_array.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_total = np.dot(np.arange(256), hist)
    sum_bg, count_bg, best_thresh, best_var = 0.0, 0.0, 128, 0.0
    for t in range(256):
        count_bg += hist[t]
        if count_bg == 0 or count_bg == total:
            continue
        count_fg = total - count_bg
        sum_bg += t * hist[t]
        mean_bg = sum_bg / count_bg
        mean_fg = (sum_total - sum_bg) / count_fg
        var = count_bg * count_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var, best_thresh = var, t

    ink_mask = img_array < best_thresh  # dark pixels = ink

    # Bounding box of all ink
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)
    if not rows.any() or not cols.any():
        region = img_array
    else:
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Shrink inward by 15% to strip the drawn frame border
        dy = int((y2 - y1) * 0.15)
        dx = int((x2 - x1) * 0.15)
        y1 = min(h - 1, y1 + dy)
        y2 = max(0, y2 - dy)
        x1 = min(w - 1, x1 + dx)
        x2 = max(0, x2 - dx)

        region = img_array[y1:y2, x1:x2]

    # Re-run Otsu on the cropped region for a tighter threshold, then binarize
    flat = region.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_total = np.dot(np.arange(256), hist)
    sum_bg2, count_bg2, best_thresh2, best_var2 = 0.0, 0.0, 128, 0.0
    for t in range(256):
        count_bg2 += hist[t]
        if count_bg2 == 0 or count_bg2 == total:
            continue
        count_fg2 = total - count_bg2
        sum_bg2 += t * hist[t]
        mean_bg2 = sum_bg2 / count_bg2
        mean_fg2 = (sum_total - sum_bg2) / count_fg2
        var2 = count_bg2 * count_fg2 * (mean_bg2 - mean_fg2) ** 2
        if var2 > best_var2:
            best_var2, best_thresh2 = var2, t

    # Binary mask: 1.0 = ink, 0.0 = background
    binary = (region < best_thresh2).astype(np.float32)
    crop = Image.fromarray((binary * 255).astype(np.uint8))

    resized = crop.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    resized_array = np.array(resized, dtype=np.float32) / 255.0

    if debug:
        print(f"Image: {Path(image_path).name}")
        print(f"Ink coverage: {resized_array.mean():.1%}")
        print("Grid (# = ink):")
        for row in resized_array[::2]:
            print(" ".join(['#' if v > 0.3 else '.' for v in row]))
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

    print("Weight zones: + = center <0.28 (positive), - = ring 0.28-0.65 (negative), . = outer")
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


# Ground truth labels for trainingData/
# Add "X" or "O" for any unlabeled images below
LABELS = {
    # from images/
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
    # from images2/
    "image1.jpeg": "O",
    "image2.jpeg": "O",
    "image3.jpeg": "X",
    "image4.jpeg": "O",
    "image5.jpeg": "X",
    "image6.jpeg": "O",
    "image7.jpeg": "O",
    "image8.jpeg": "X",
    "image9.jpeg": "X",
    "image10.jpeg": "O",
    # from images3/
    "IMG_3240.jpg": "X",
    "IMG_3241.jpg": "O",
    "IMG_3242.jpg": "X",
    "IMG_3243.jpg": "O",
    "IMG_3244.jpg": "O",
    "IMG_3245.jpg": "O",
    "IMG_3246.jpg": "X",
    "IMG_3247.jpg": "X",
    "IMG_3248.jpg": "O",
    "IMG_3249.jpg": "X",
    "IMG_3250.jpg": "X",
    "IMG_3251.jpg": "X",
    "IMG_3252.jpg": "X",
    "IMG_3253.jpg": "O",
    # from images4/
    "IMG_3263.jpg": "X",
    "IMG_3264.jpg": "O",
    "IMG_3265.jpg": "X",
    "IMG_3266.jpg": "O",
    "IMG_3267.jpg": "X",
    "IMG_3268.jpg": "X",
    "IMG_3269.jpg": "O",
    "IMG_4244.jpg": "O",
    "IMG_4245.jpg": "X",
    "IMG_4246.jpg": "X",
    "IMG_4247.jpg": "X",
    "IMG_4248.jpg": "O",
    "IMG_4249.jpg": "O",
    "IMG_4250.jpg": "X",
    "IMG_4251.jpg": "O",
    "IMG_4252.jpg": "O",
    "IMG_4253.jpg": "O",
    "IMG_4254.jpg": "X",
    "IMG_4255.jpg": "X",
    "IMG_4256.jpg": "O",
    "IMG_4257.jpg": "O",
    "IMG_4258.jpg": "X",
    "IMG_4259.jpg": "X",
    "IMG_4260.jpg": "O",
    "IMG_4261.jpg": "O",
    "IMG_4262.jpg": "O",
    "IMG_4263.jpg": "X",
    "IMG_4264.jpg": "X",
    "img1.jpeg": "X",
    "img2.jpeg": "O",
    "img3.jpeg": "X",
    "img4.jpeg": "O",
    "img5.jpeg": "X",
    "img6.jpeg": "O",
    "img7.jpeg": "X",
    "img8.jpeg": "O",
    "img9.jpeg": "O",
    "img10.jpeg": "O",
    # from testSet2/
    "IMG_3270_2.jpg": "O",
    "IMG_3271_2.jpg": "X",
    "IMG_3272_2.jpg": "X",
    "IMG_3273_2.jpg": "O",
    "IMG_3274_2.jpg": "O",
    "IMG_3275_2.jpg": "X",
    "IMG_3276_2.jpg": "O",
    "IMG_3277_2.jpg": "X",
    "IMG_3278_2.jpg": "X",
    "IMG_3279_2.jpg": "O",
    "IMG_3280_2.jpg": "X",
    "IMG_3281_2.jpg": "X",
    "IMG_3390.jpg": "X",
    "IMG_3391.jpg": "X",
    "IMG_3392.jpg": "X",
    "IMG_3393.jpg": "X",
    "IMG_3394.jpg": "X",
    "IMG_3395.jpg": "O",
    "IMG_3396.jpg": "O",
    "IMG_3397.jpg": "O",
    "IMG_3398.jpg": "O",
    "IMG_3399.jpg": "O",
    "IMG_3400.jpg": "X",
    "IMG_3401.jpg": "X",
    "IMG_3402.jpg": "X",
    "IMG_3403.jpg": "X",
    "IMG_3404.jpg": "X",
    "IMG_3405.jpg": "O",
    "IMG_3406.jpg": "O",
    "IMG_3407.jpg": "O",
    "IMG_3408.jpg": "O",
    "IMG_3409.jpg": "O",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify X vs O images using a manually crafted perceptron")
    parser.add_argument("-d", "--dir", type=str, default="trainingData",
                        help="Directory containing images to classify (default: trainingData)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed debug output for each image")
    parser.add_argument("--predict", type=str, default=None,
                        help="Predict a single image file and exit")
    args = parser.parse_args()

    # Single-image prediction mode
    if args.predict:
        img_path = Path(args.predict)
        if not img_path.exists():
            print(f"Error: File not found: {img_path}")
            exit(1)
        label, confidence, _ = classify_image(img_path)
        print(f"Prediction: {label} (confidence: {confidence:.1%})")
        exit(0)

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
