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


def find_frame_bounds(img_array, debug=False):
    """
    Detect the rectangular frame in the image and return its inner bounds.

    Returns (y1, y2, x1, x2) for the region inside the frame.
    """
    h, w = img_array.shape

    # Invert so ink = bright
    inverted = 255 - img_array

    # Threshold to find ink pixels (top 10% brightness)
    threshold = np.percentile(inverted, 90)
    ink_mask = inverted > threshold

    # Find rows and columns that contain significant ink
    row_ink = np.sum(ink_mask, axis=1)
    col_ink = np.sum(ink_mask, axis=0)

    # Find the frame boundaries by looking for rows/cols with ink
    # The frame should be the outermost ink lines
    row_threshold = w * 0.02  # At least 2% of width has ink
    col_threshold = h * 0.02  # At least 2% of height has ink

    rows_with_ink = np.where(row_ink > row_threshold)[0]
    cols_with_ink = np.where(col_ink > col_threshold)[0]

    if len(rows_with_ink) < 2 or len(cols_with_ink) < 2:
        # Fallback to center crop if frame detection fails
        margin_h = int(h * 0.30)
        margin_w = int(w * 0.30)
        return margin_h, h - margin_h, margin_w, w - margin_w

    # Frame boundaries (outer edges of the frame)
    frame_top = rows_with_ink[0]
    frame_bottom = rows_with_ink[-1]
    frame_left = cols_with_ink[0]
    frame_right = cols_with_ink[-1]

    # Calculate frame dimensions
    frame_height = frame_bottom - frame_top
    frame_width = frame_right - frame_left

    # Crop to INSIDE the frame (add margin to exclude frame lines)
    margin = 0.1  # 10% margin inside the frame
    y1 = int(frame_top + frame_height * margin)
    y2 = int(frame_bottom - frame_height * margin)
    x1 = int(frame_left + frame_width * margin)
    x2 = int(frame_right - frame_width * margin)

    # Ensure valid bounds
    y1 = max(0, y1)
    y2 = min(h, y2)
    x1 = max(0, x1)
    x2 = min(w, x2)

    if debug:
        print(f"Frame detected: top={frame_top}, bottom={frame_bottom}, left={frame_left}, right={frame_right}")
        print(f"Inner region: y=[{y1}:{y2}], x=[{x1}:{x2}]")

    return y1, y2, x1, x2


def preprocess_image(image_path, grid_size=16, debug=False):
    """
    Preprocess image: detect frame, crop inside it, center on ink, downsample.
    """
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    h, w = img_array.shape

    # Detect frame and get inner bounds
    y1, y2, x1, x2 = find_frame_bounds(img_array, debug=debug)

    # Crop to inside the frame
    cropped = img_array[y1:y2, x1:x2]

    # Invert so ink = bright
    inverted = 255 - cropped

    # Find ink using threshold
    threshold = np.percentile(inverted, 88)
    ink_mask = (inverted > threshold).astype(np.float32)

    # Find centroid of ink to center the shape
    ink_coords = np.where(ink_mask > 0)
    if len(ink_coords[0]) > 0:
        cy = int(np.mean(ink_coords[0]))
        cx = int(np.mean(ink_coords[1]))
    else:
        cy, cx = ink_mask.shape[0] // 2, ink_mask.shape[1] // 2

    # Extract a square region centered on the ink centroid
    ch, cw = ink_mask.shape
    region_size = min(ch, cw) * 0.7  # 70% of the smaller dimension

    half_size = int(region_size // 2)
    cy1 = max(0, cy - half_size)
    cy2 = min(ch, cy + half_size)
    cx1 = max(0, cx - half_size)
    cx2 = min(cw, cx + half_size)

    # Crop centered on ink
    centered_ink = ink_mask[cy1:cy2, cx1:cx2]

    # Resize to grid_size x grid_size
    ink_img = Image.fromarray((centered_ink * 255).astype(np.uint8))
    resized = ink_img.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    resized_array = np.array(resized, dtype=np.float32) / 255.0

    if debug:
        print(f"Image: {Path(image_path).name}")
        print(f"Ink centroid: ({cx}, {cy}), Region: [{cx1}:{cx2}, {cy1}:{cy2}]")
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
    # images2/ directory
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
    # images3/ directory
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
