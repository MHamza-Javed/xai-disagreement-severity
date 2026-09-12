"""
Data Loading and Preprocessing Module
- Load MVTec AD dataset (hazelnut, screw)
- Split defective images 30/20/50 (train/val/test), stratified per defect subtype
- Split native test/good images into val/test using the same val:test ratio
- Resize images to 224x224, save preprocessed copies + aligned ground-truth masks
"""

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

RANDOM_SEED = 42
IMG_SIZE = 224

# Split ratios (locked in Phase 1 decision)
DEFECTIVE_SPLIT = {"train": 0.30, "val": 0.20, "test": 0.50}
# Good test images use the same val:test proportion (no train split needed - train/good already exists)
GOOD_TEST_VAL_RATIO = 0.20 / (0.20 + 0.50)  # ≈ 0.286


def _list_images(dir_path):
    """Return sorted list of image file paths in a directory."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


def build_splits(mvtec_root, category, seed=RANDOM_SEED):
    """
    Build train/val/test splits for one MVTec AD category.

    Returns:
        dict: {
            'train': [(img_path, mask_path_or_None, label, defect_type), ...],
            'val':   [...],
            'test':  [...],
        }
        label: 0 = good, 1 = defective
    """
    random.seed(seed)
    cat_root = Path(mvtec_root) / category
    splits = {"train": [], "val": [], "test": []}

    # --- GOOD images ---
    train_good = _list_images(cat_root / "train" / "good")
    for p in train_good:
        splits["train"].append((p, None, 0, "good"))

    test_good = _list_images(cat_root / "test" / "good")
    random.shuffle(test_good)
    n_val_good = int(len(test_good) * GOOD_TEST_VAL_RATIO)
    for p in test_good[:n_val_good]:
        splits["val"].append((p, None, 0, "good"))
    for p in test_good[n_val_good:]:
        splits["test"].append((p, None, 0, "good"))

    # --- DEFECTIVE images (stratified per defect subtype) ---
    test_dir = cat_root / "test"
    gt_dir = cat_root / "ground_truth"
    defect_types = [d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]

    for defect_type in defect_types:
        imgs = _list_images(test_dir / defect_type)
        random.shuffle(imgs)
        n = len(imgs)
        n_train = round(n * DEFECTIVE_SPLIT["train"])
        n_val = round(n * DEFECTIVE_SPLIT["val"])
        # remainder goes to test to avoid rounding loss

        buckets = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }

        if n < 15:
            print(f"  ⚠️  WARNING: {category}/{defect_type} has only {n} images "
                  f"— train split will have ~{n_train} images, check quality manually.")

        for split_name, split_imgs in buckets.items():
            for img_path in split_imgs:
                mask_path = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                mask_path = mask_path if mask_path.exists() else None
                splits[split_name].append((img_path, mask_path, 1, defect_type))

    return splits


def _resize_and_save(src_path, dst_path, size=IMG_SIZE, is_mask=False):
    """Resize an image (or mask) and save to destination."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {src_path}")
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (size, size), interpolation=interp)
    cv2.imwrite(str(dst_path), resized)


def preprocess_category(category, mvtec_root=None, preprocessed_root=None, masks_root=None):
    """
    Run full preprocessing for one category: build splits, resize, save.
    Populates data/preprocessed/{category}/ and data/ground_truth_masks/{category}/.
    """
    mvtec_root = mvtec_root or config.MVTEC_AD_PATH
    preprocessed_root = preprocessed_root or config.PREPROCESSED_PATH
    masks_root = masks_root or (config.DATA_ROOT / "ground_truth_masks")

    print(f"\n📦 Preprocessing category: {category}")
    splits = build_splits(mvtec_root, category)

    total_images = sum(len(items) for items in splits.values())
    print(f"  Total images to process: {total_images}")

    manifest = []  # track every processed file for later loading

    for split_name, items in splits.items():
        label_counts = {"good": 0, "def": 0}

        for img_path, mask_path, label, defect_type in tqdm(items, desc=f"  {split_name}", unit="img"):
            label_str = "good" if label == 0 else "def"
            label_counts[label_str] += 1
            idx = label_counts[label_str]

            short_cat = category[:5]  # e.g. "hazel", "screw"
            fname = f"{short_cat}_{split_name}_{label_str}_{idx:03d}.png"

            dst_img = preprocessed_root / category / split_name / label_str / fname
            _resize_and_save(img_path, dst_img, is_mask=False)

            dst_mask = None
            if mask_path is not None:
                mask_fname = fname.replace(".png", "_mask.png")
                dst_mask = masks_root / category / split_name / mask_fname
                _resize_and_save(mask_path, dst_mask, is_mask=True)

            manifest.append({
                "image_path": str(dst_img),
                "mask_path": str(dst_mask) if dst_mask else None,
                "label": label,
                "split": split_name,
                "category": category,
                "defect_type": defect_type,
            })

        print(f"  ✅ {split_name}: {label_counts['good']} good, {label_counts['def']} defective")

    # Save manifest as CSV for easy reloading
    import csv
    manifest_path = preprocessed_root / category / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"  📄 Manifest saved: {manifest_path}")

    return manifest


class MVTecADDataset(Dataset):
    """
    Loads preprocessed MVTec AD images from a manifest CSV.

    Args:
        manifest_path (str): Path to manifest.csv (from preprocess_category)
        split (str): 'train', 'val', or 'test'
        transform: torchvision transform (default: ImageNet normalization)
    """

    def __init__(self, manifest_path, split, transform=None):
        import csv
        self.items = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.items.append(row)

        category = self.items[0]["category"] if self.items else None

        if transform is not None:
            self.transform = transform
        elif split == "train" and category == "hazelnut":
            # Hazelnuts are roughly round/symmetric — flips and rotation are realistic
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif split == "train" and category == "screw":
            # Screws are orientation-sensitive (thread direction, head-up view) —
            # only mild rotation, no vertical flip (would invert thread direction unrealistically)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(item["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        label = int(item["label"])

        mask = None
        if item["mask_path"]:
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            mask = torch.from_numpy(mask)

        return img, label, mask, item["image_path"]


def get_mvtec_loaders(manifest_path, batch_size=32, num_workers=4):
    """Create train/val/test DataLoaders from a manifest."""
    def collate(batch):
        imgs, labels, masks, paths = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        return imgs, labels, masks, paths  # masks kept as list (variable presence)

    train_ds = MVTecADDataset(manifest_path, "train")
    val_ds = MVTecADDataset(manifest_path, "val")
    test_ds = MVTecADDataset(manifest_path, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    for category in config.MVTEC_CATEGORIES:
        preprocess_category(category)
    print("\n✨ Preprocessing complete for all categories.")