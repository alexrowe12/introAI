"""
Classifier 2: Multi-Layer Perceptron (MLP) for X vs O Classification

This classifier is TRAINED on your dataset, unlike Classifier 1.
Uses the same preprocessing (frame detection, centering) as Classifier 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import random


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for X vs O classification.

    Architecture:
    - Input: flattened image (grid_size * grid_size)
    - Hidden layer 1: 64 neurons with ReLU
    - Hidden layer 2: 32 neurons with ReLU
    - Output: 1 neuron (sigmoid for binary classification)
    """

    def __init__(self, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        input_size = grid_size * grid_size

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            prob = self.forward(x)
            return (prob > 0.5).int().squeeze()

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x).squeeze()


# Ground truth labels (same as classifier1)
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
}


def find_frame_bounds(img_array, debug=False):
    """
    Detect the rectangular frame in the image and return its inner bounds.
    Returns (y1, y2, x1, x2) for the region inside the frame.
    """
    h, w = img_array.shape
    inverted = 255 - img_array
    threshold = np.percentile(inverted, 90)
    ink_mask = inverted > threshold

    row_ink = np.sum(ink_mask, axis=1)
    col_ink = np.sum(ink_mask, axis=0)

    row_threshold = w * 0.02
    col_threshold = h * 0.02

    rows_with_ink = np.where(row_ink > row_threshold)[0]
    cols_with_ink = np.where(col_ink > col_threshold)[0]

    if len(rows_with_ink) < 2 or len(cols_with_ink) < 2:
        margin_h = int(h * 0.30)
        margin_w = int(w * 0.30)
        return margin_h, h - margin_h, margin_w, w - margin_w

    frame_top = rows_with_ink[0]
    frame_bottom = rows_with_ink[-1]
    frame_left = cols_with_ink[0]
    frame_right = cols_with_ink[-1]

    frame_height = frame_bottom - frame_top
    frame_width = frame_right - frame_left

    margin = 0.1
    y1 = int(frame_top + frame_height * margin)
    y2 = int(frame_bottom - frame_height * margin)
    x1 = int(frame_left + frame_width * margin)
    x2 = int(frame_right - frame_width * margin)

    y1 = max(0, y1)
    y2 = min(h, y2)
    x1 = max(0, x1)
    x2 = min(w, x2)

    return y1, y2, x1, x2


def preprocess_image(image_path, grid_size=16, augment=False):
    """
    Preprocess image: detect frame, crop inside it, center on ink, downsample.

    If augment=True, applies random transformations for data augmentation.
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    y1, y2, x1, x2 = find_frame_bounds(img_array)
    cropped = img_array[y1:y2, x1:x2]

    inverted = 255 - cropped
    threshold = np.percentile(inverted, 88)
    ink_mask = (inverted > threshold).astype(np.float32)

    # Find centroid of ink
    ink_coords = np.where(ink_mask > 0)
    if len(ink_coords[0]) > 0:
        cy = int(np.mean(ink_coords[0]))
        cx = int(np.mean(ink_coords[1]))
    else:
        cy, cx = ink_mask.shape[0] // 2, ink_mask.shape[1] // 2

    ch, cw = ink_mask.shape
    region_size = min(ch, cw) * 0.7

    # Apply augmentation: random offset to centroid
    if augment:
        offset_range = int(region_size * 0.1)
        cy += random.randint(-offset_range, offset_range)
        cx += random.randint(-offset_range, offset_range)

    half_size = int(region_size // 2)
    cy1 = max(0, cy - half_size)
    cy2 = min(ch, cy + half_size)
    cx1 = max(0, cx - half_size)
    cx2 = min(cw, cx + half_size)

    centered_ink = ink_mask[cy1:cy2, cx1:cx2]

    ink_img = Image.fromarray((centered_ink * 255).astype(np.uint8))

    # Apply augmentation: random rotation and scaling
    if augment:
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.9, 1.1)
        new_size = int(ink_img.width * scale), int(ink_img.height * scale)
        ink_img = ink_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        ink_img = ink_img.resize(new_size, Image.Resampling.BILINEAR)
        # Center crop back to original size
        w, h = ink_img.size
        target_w, target_h = int(centered_ink.shape[1]), int(centered_ink.shape[0])
        left = max(0, (w - target_w) // 2)
        top = max(0, (h - target_h) // 2)
        ink_img = ink_img.crop((left, top, left + target_w, top + target_h))

    resized = ink_img.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    resized_array = np.array(resized, dtype=np.float32) / 255.0

    return torch.tensor(resized_array.flatten(), dtype=torch.float32)


class XODataset(Dataset):
    """Dataset for X vs O images with precomputed features."""

    def __init__(self, image_dirs, labels_dict, grid_size=16, augment=False, augment_factor=10):
        """
        Args:
            image_dirs: List of directories containing images
            labels_dict: Dictionary mapping filename to label ("X" or "O")
            grid_size: Size to resize images to
            augment: Whether to apply data augmentation
            augment_factor: How many augmented copies per original image
        """
        self.grid_size = grid_size

        # Precompute all features at initialization
        self.features_list = []
        self.labels_list = []

        samples = []
        for image_dir in image_dirs:
            image_dir = Path(image_dir)
            if not image_dir.exists():
                continue
            image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
            for img_path in image_files:
                if img_path.name in labels_dict:
                    label = 1 if labels_dict[img_path.name] == "X" else 0
                    samples.append((img_path, label))

        print(f"Loading {len(samples)} images...", flush=True)

        # Precompute original features
        for img_path, label in samples:
            features = preprocess_image(img_path, grid_size, augment=False)
            self.features_list.append(features)
            self.labels_list.append(label)

        # Add augmented copies
        if augment:
            print(f"Generating {augment_factor-1}x augmented copies...", flush=True)
            for _ in range(augment_factor - 1):
                for img_path, label in samples:
                    features = preprocess_image(img_path, grid_size, augment=True)
                    self.features_list.append(features)
                    self.labels_list.append(label)

        print(f"Total training samples: {len(self.features_list)}", flush=True)

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        return self.features_list[idx], torch.tensor(self.labels_list[idx], dtype=torch.float32)


def train_model(model, train_loader, epochs=100, lr=0.01, verbose=True):
    """Train the MLP model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if verbose and (epoch + 1) % 10 == 0:
            accuracy = correct / total
            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Accuracy={accuracy:.1%}", flush=True)

    return model


def evaluate_model(model, image_dirs, labels_dict, grid_size=16, verbose=True):
    """Evaluate the model on images."""
    model.eval()
    results = []
    correct = 0
    total = 0

    for image_dir in image_dirs:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            continue

        if verbose:
            print(f"\nEvaluating on {image_dir.name}/:")
            print("-" * 40)

        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
        for img_path in sorted(image_files):
            if img_path.name not in labels_dict:
                continue

            features = preprocess_image(img_path, grid_size, augment=False)
            features = features.unsqueeze(0)

            prob = model.predict_proba(features).item()
            pred = 1 if prob > 0.5 else 0
            pred_label = "X" if pred == 1 else "O"
            true_label = labels_dict[img_path.name]
            is_correct = pred_label == true_label

            confidence = prob if pred == 1 else 1 - prob

            results.append({
                "file": img_path.name,
                "prediction": pred_label,
                "true_label": true_label,
                "confidence": confidence,
                "correct": is_correct
            })

            if is_correct:
                correct += 1
            total += 1

            if verbose:
                status = " ok" if is_correct else f" WRONG (true: {true_label})"
                print(f"{img_path.name}: {pred_label} (conf: {confidence:.1%}){status}")

    accuracy = correct / total if total > 0 else 0
    if verbose:
        print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.1%}")

    return {"results": results, "accuracy": accuracy, "correct": correct, "total": total}


def save_model(model, path):
    """Save model weights to file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'grid_size': model.grid_size
    }, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load model weights from file."""
    checkpoint = torch.load(path, weights_only=True)
    model = MLP(grid_size=checkpoint['grid_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test MLP classifier for X vs O")
    parser.add_argument("-d", "--dir", type=str, nargs="+", default=["images", "images2"],
                        help="Directories containing images (default: images images2)")
    parser.add_argument("--train", action="store_true",
                        help="Train a new model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--model", type=str, default="mlp_model.pth",
                        help="Path to save/load model (default: mlp_model.pth)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    print("=" * 60)
    print("Classifier 2: Multi-Layer Perceptron (MLP)")
    print("=" * 60)
    print()

    # Resolve image directories
    script_dir = Path(__file__).parent
    image_dirs = []
    for d in args.dir:
        dir_path = Path(d)
        if not dir_path.is_absolute():
            dir_path = script_dir / d
        if dir_path.exists():
            image_dirs.append(dir_path)
        else:
            print(f"Warning: Directory not found: {dir_path}")

    if not image_dirs:
        print("Error: No valid image directories found!")
        exit(1)

    print(f"Image directories: {[str(d) for d in image_dirs]}")
    print()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = script_dir / args.model

    if args.train:
        print("TRAINING MODE")
        print("-" * 40)

        # Create dataset with augmentation
        dataset = XODataset(image_dirs, LABELS, grid_size=16, augment=True, augment_factor=10)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Create and train model
        model = MLP(grid_size=16)
        print(f"\nTraining for {args.epochs} epochs...")
        print()
        model = train_model(model, train_loader, epochs=args.epochs, verbose=True)

        # Save model
        save_model(model, model_path)
        print()
    else:
        # Load existing model or train new one
        if model_path.exists():
            model = load_model(model_path)
        else:
            print(f"No saved model found at {model_path}")
            print("Training a new model...")
            print()

            dataset = XODataset(image_dirs, LABELS, grid_size=16, augment=True, augment_factor=10)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

            model = MLP(grid_size=16)
            model = train_model(model, train_loader, epochs=50, verbose=True)
            save_model(model, model_path)
            print()

    print()
    print("EVALUATION")
    print("-" * 40)
    results = evaluate_model(model, image_dirs, LABELS, grid_size=16, verbose=True)

    print()
    print("=" * 60)
    print("USAGE:")
    print("  python classifier2_mlp.py                    # Evaluate with saved model")
    print("  python classifier2_mlp.py --train            # Train new model")
    print("  python classifier2_mlp.py --train --epochs 200")
    print("  python classifier2_mlp.py -d images          # Evaluate on specific directory")
    print("=" * 60)
