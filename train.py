"""
Training Script — Phase 1
Fine-tunes ResNet-18 and ViT on both hazelnut and screw categories.

Usage:
    python train.py
"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
import config
from src.data_loader import get_mvtec_loaders
from src.model_utils import (
    ResNet18Classifier, ViTClassifier,
    compute_class_weights, train_model,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for category in config.MVTEC_CATEGORIES:
        manifest_path = config.PREPROCESSED_PATH / category / "manifest.csv"

        if not manifest_path.exists():
            print(f"⚠️  Manifest not found for {category}: {manifest_path}")
            print("   Run 'python src/data_loader.py' first.")
            continue

        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader = get_mvtec_loaders(
            manifest_path, batch_size=config.BATCH_SIZE, num_workers=0  # 0 workers avoids Windows multiprocessing issues
        )

        print(f"  Train: {len(train_loader.dataset)} images")
        print(f"  Val:   {len(val_loader.dataset)} images")
        print(f"  Test:  {len(test_loader.dataset)} images")

        # Compute class weights once per category (shared across both architectures)
        print("  Computing class weights for imbalance handling...")
        class_weights = compute_class_weights(train_loader, num_classes=2, device=device)

        # --- Train ResNet-18 ---
        resnet_model = ResNet18Classifier(pretrained=True, num_classes=2)
        train_model(
            model=resnet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            device=device,
            model_name="resnet18",
            save_dir=config.MODELS_PATH / "resnet18" / category,
            class_weights=class_weights,
        )

         # --- Train ViT ---
        # Backbone kept frozen for the full run (warmup_epochs = num_epochs):
        # unfreezing destabilized training (optimizer state reset + head/backbone
        # mismatch) without improving accuracy. Using pretrained ViT features
        # as-is (linear probe) is more stable for this dataset size.
        vit_model = ViTClassifier(pretrained=True, num_classes=2)
        train_model(
            model=vit_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            device=device,
            model_name="vit",
            save_dir=config.MODELS_PATH / "vit" / category,
            class_weights=class_weights,
            warmup_epochs=config.NUM_EPOCHS,  # never unfreeze — frozen backbone throughout
        )

    print("\n✨ All training complete!")


if __name__ == "__main__":
    main()