"""
Classifier 2: Multi-Layer Perceptron (MLP) for X vs O Classification

This classifier is TRAINED on your dataset, unlike Classifier 1.
Uses the same preprocessing (frame detection, centering) as Classifier 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
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
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
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
}


def preprocess_image(image_path, grid_size=16, augment=False):
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

    if augment:
        angle = random.uniform(-15, 15)
        crop = crop.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    resized = crop.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
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


def train_model(model, train_loader, epochs=100, lr=0.01, verbose=True, device=None):
    """Train the MLP model."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

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


def evaluate_model(model, image_dirs, labels_dict, grid_size=16, verbose=True, device=None):
    """Evaluate the model on images."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
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
            features = features.unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(features).squeeze().item()
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
    parser.add_argument("-d", "--dir", type=str, nargs="+", default=["trainingData"],
                        help="Directories containing images (default: images images2)")
    parser.add_argument("--train", action="store_true",
                        help="Train a new model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--model", type=str, default="mlp_model.pth",
                        help="Path to save/load model (default: mlp_model.pth)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration (MPS on Apple Silicon)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Predict a single image file and exit")
    args = parser.parse_args()

    # Set up device
    if args.gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using GPU: Apple MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: CUDA")
        else:
            print("Warning: GPU requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Single-image prediction mode
    if args.predict:
        img_path = Path(args.predict)
        if not img_path.exists():
            print(f"Error: File not found: {img_path}")
            exit(1)
        script_dir = Path(__file__).parent
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = script_dir / args.model
        if not model_path.exists():
            print(f"Error: No model found at {model_path}. Train first with --train.")
            exit(1)
        model = load_model(model_path)
        model.eval()
        features = preprocess_image(img_path, grid_size=model.grid_size)
        features = features.unsqueeze(0)
        with torch.no_grad():
            prob = model(features).squeeze().item()
        pred = "X" if prob > 0.5 else "O"
        confidence = prob if prob > 0.5 else 1 - prob
        print(f"Prediction: {pred} (confidence: {confidence:.1%})")
        exit(0)

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
        model = train_model(model, train_loader, epochs=args.epochs, verbose=True, device=device)

        # Save model (move to CPU first for compatibility)
        model = model.to("cpu")
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
            model = train_model(model, train_loader, epochs=50, verbose=True, device=device)
            model = model.to("cpu")
            save_model(model, model_path)
            print()

    print()
    print("EVALUATION")
    print("-" * 40)
    results = evaluate_model(model, image_dirs, LABELS, grid_size=16, verbose=True, device=device)

    print()
    print("=" * 60)
    print("USAGE:")
    print("  python classifier2_mlp.py                    # Evaluate with saved model")
    print("  python classifier2_mlp.py --train            # Train new model")
    print("  python classifier2_mlp.py --train --gpu      # Train using GPU (Apple MPS)")
    print("  python classifier2_mlp.py --train --epochs 200")
    print("  python classifier2_mlp.py -d images          # Evaluate on specific directory")
    print("=" * 60)
