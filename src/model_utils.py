"""
Model Training and Evaluation
- Fine-tune ResNet-18 for MVTec AD
- Fine-tune Vision Transformer (ViT) for MVTec AD
- Handles class imbalance via weighted cross-entropy loss
- Differential learning rates (head vs. backbone) for stable ViT fine-tuning
- Gradient clipping for training stability
- Model checkpoint saving/loading, training logs
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import timm


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 fine-tuned for binary anomaly detection (good vs. defective).
    """

    def __init__(self, pretrained=True, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


class ViTClassifier(nn.Module):
    """
    Vision Transformer (small) fine-tuned for binary anomaly detection.
    """

    def __init__(self, pretrained=True, num_classes=2):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


def compute_class_weights(train_loader, num_classes=2, device='cpu'):
    """
    Compute inverse-frequency class weights from a DataLoader for weighted CE loss.

    Returns:
        weights (Tensor): [num_classes] weight per class
    """
    counts = torch.zeros(num_classes)
    for _, labels, _, _ in train_loader:
        for c in range(num_classes):
            counts[c] += (labels == c).sum().item()

    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    print(f"  Class counts: {counts.tolist()} -> weights: {weights.tolist()}")
    return weights.to(device)


def build_optimizer(model, lr, head_lr_multiplier=10.0, weight_decay=1e-4):
    """
    Build an optimizer with a higher learning rate for the classification head
    (randomly initialized) vs. the pretrained backbone. This stabilizes ViT
    fine-tuning on small datasets, where a single low LR either leaves the head
    undertrained or destabilizes the backbone.
    """
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if "fc." in name or name.startswith("model.fc") or ".head." in name or name.startswith("model.head"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return optim.Adam([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_multiplier},
    ], weight_decay=weight_decay)


def set_backbone_trainable(model, trainable):
    """
    Freeze or unfreeze all parameters except the classification head.
    Used for two-stage training: head-only warmup, then full fine-tuning.
    """
    for name, param in model.named_parameters():
        is_head = ("fc." in name or name.startswith("model.fc")
                   or ".head." in name or name.startswith("model.head"))
        if not is_head:
            param.requires_grad = trainable


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Returns:
        avg_loss (float), accuracy (float)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _, _ in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate model on validation/test set.

    Returns:
        avg_loss (float), accuracy (float)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint.get('val_acc', None)
    print(f"Checkpoint loaded from {path} (epoch {epoch}, val_acc {val_acc})")
    return model, optimizer, epoch


def train_model(model, train_loader, val_loader, num_epochs, lr, device,
                 model_name, save_dir, class_weights=None, warmup_epochs=3):
    """
    Full training loop with two-stage training and checkpoint saving.

    Stage 1 (warmup_epochs): backbone frozen, only classification head trains.
        This gives the randomly-initialized head a sane starting point before
        it's allowed to influence the pretrained backbone at all — prevents
        the early "yanking" that destabilizes ViT fine-tuning on small datasets.
    Stage 2 (remaining epochs): full model unfrozen, differential LR
        (backbone at `lr`, head at `lr * 10`) for joint fine-tuning.

    Args:
        model (nn.Module): ResNet18Classifier or ViTClassifier
        train_loader, val_loader: DataLoaders from get_mvtec_loaders
        num_epochs (int): Total training epochs (including warmup)
        lr (float): Base learning rate (backbone); head uses lr * 10
        device (str): 'cuda' or 'cpu'
        model_name (str): 'resnet18' or 'vit' (used for filenames)
        save_dir (str or Path): Directory to save checkpoints + logs
        class_weights (Tensor or None): Weights for CrossEntropyLoss (handles imbalance)
        warmup_epochs (int): Number of head-only epochs before unfreezing backbone

    Returns:
        model (trained), log_rows (list of dicts)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    log_rows = []

    print(f"\n🚀 Training {model_name} for {num_epochs} epochs on {device} "
          f"({warmup_epochs} warmup epochs, backbone frozen)...")

    # Stage 1: freeze backbone, warmup head
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer(model, lr=lr, head_lr_multiplier=10.0, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        # Stage 2: unfreeze backbone after warmup, rebuild optimizer
        if epoch == warmup_epochs + 1:
            print(f"  🔓 Unfreezing backbone at epoch {epoch}")
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model, lr=lr, head_lr_multiplier=10.0, weight_decay=1e-4)

        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start

        stage_tag = "warmup" if epoch <= warmup_epochs else "finetune"
        print(f"  Epoch {epoch:3d}/{num_epochs} [{stage_tag}] | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed_sec": elapsed,
        })

        # Save best checkpoint: prioritize val_acc, use val_loss as tie-breaker
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_acc,
                             save_dir / f"{model_name}_best_val_acc.pth")

    # Save final checkpoint (last epoch, regardless of val_acc)
    save_checkpoint(model, optimizer, num_epochs, val_acc, save_dir / f"{model_name}_final.pth")

    # Save training log as CSV
    log_path = save_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"✅ {model_name} training complete. Best val_acc: {best_val_acc:.4f}")
    print(f"   Checkpoints saved to: {save_dir}")

    return model, log_rows