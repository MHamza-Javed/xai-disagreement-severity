"""
Phase 1 Closeout — Test Set Evaluation & Filtering
- Loads *_best_val_acc.pth checkpoints (NOT _final.pth) for all 4 combinations
- Evaluates on the held-out test set
- Reports overall accuracy, per-class accuracy, and per-defect-type correct counts
- Saves a filtered manifest (correctly-classified images only) per category+architecture
  — this filtered set is what Phase 2 XAI generation will use

Usage:
    python evaluate_models.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import torch

sys.path.append(str(Path(__file__).parent))
import config
from src.data_loader import MVTecADDataset
from src.model_utils import ResNet18Classifier, ViTClassifier, load_checkpoint

MIN_RECOMMENDED_DEFECTIVE = 15  # threshold flagged in the critique for a "usable" XAI sample


def evaluate_on_test(model, test_dataset, device):
    """
    Run model on every test image individually (not batched, so we can
    track per-image results including defect_type for later analysis).

    Returns:
        results (list of dict): one row per test image with prediction info
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, label, mask, img_path = test_dataset[i]
            item = test_dataset.items[i]

            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = output.argmax(dim=1).item()

            results.append({
                "image_path": img_path,
                "true_label": label,
                "pred_label": pred,
                "correct": int(pred == label),
                "defect_type": item["defect_type"],
                "mask_path": item["mask_path"],
            })

    return results


def summarize_results(results, category, arch_name):
    """Print accuracy breakdown and return summary dict."""
    total = len(results)
    correct = sum(r["correct"] for r in results)
    overall_acc = correct / total if total else 0.0

    # Per-class accuracy
    good_results = [r for r in results if r["true_label"] == 0]
    def_results = [r for r in results if r["true_label"] == 1]

    good_acc = sum(r["correct"] for r in good_results) / len(good_results) if good_results else 0.0
    def_acc = sum(r["correct"] for r in def_results) / len(def_results) if def_results else 0.0

    # Per-defect-type correct counts (only among defective images)
    defect_type_counts = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in def_results:
        dt = r["defect_type"]
        defect_type_counts[dt]["total"] += 1
        defect_type_counts[dt]["correct"] += r["correct"]

    n_correct_defective = sum(r["correct"] for r in def_results)

    print(f"\n  📊 {category} / {arch_name} — Test Set Results")
    print(f"     Overall accuracy:    {overall_acc:.4f} ({correct}/{total})")
    print(f"     Good-class accuracy: {good_acc:.4f} ({sum(r['correct'] for r in good_results)}/{len(good_results)})")
    print(f"     Defect-class accuracy: {def_acc:.4f} ({n_correct_defective}/{len(def_results)})")
    print(f"     Per-defect-type breakdown:")
    for dt, counts in sorted(defect_type_counts.items()):
        print(f"       - {dt:20s}: {counts['correct']:3d}/{counts['total']:3d} correct")

    flag = ""
    if n_correct_defective < MIN_RECOMMENDED_DEFECTIVE:
        flag = f"  ⚠️  WARNING: only {n_correct_defective} correctly-classified defective images " \
               f"(recommended minimum: {MIN_RECOMMENDED_DEFECTIVE}) — XAI sample may be too small!"
        print(flag)
    else:
        print(f"     ✅ {n_correct_defective} correctly-classified defective images — sufficient for XAI analysis")

    return {
        "category": category,
        "architecture": arch_name,
        "overall_acc": overall_acc,
        "good_acc": good_acc,
        "defective_acc": def_acc,
        "n_correct_defective": n_correct_defective,
        "n_total_defective": len(def_results),
        "flagged_low_sample": n_correct_defective < MIN_RECOMMENDED_DEFECTIVE,
    }


def save_filtered_manifest(results, category, arch_name, output_dir):
    """Save only correctly-classified images as a manifest for Phase 2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_results = [r for r in results if r["correct"] == 1]

    manifest_path = output_dir / f"{category}_{arch_name}_correctly_classified.csv"
    with open(manifest_path, "w", newline="") as f:
        fieldnames = ["image_path", "true_label", "pred_label", "defect_type", "mask_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in correct_results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"     💾 Filtered manifest saved: {manifest_path} ({len(correct_results)} images)")
    return manifest_path


def main():
    device = "cpu"  # inference is cheap; no need for GPU check here
    print(f"Using device: {device}")

    all_summaries = []
    output_dir = config.RESULTS_PATH / "metrics" / "phase1_evaluation"

    for category in config.MVTEC_CATEGORIES:
        manifest_path = config.PREPROCESSED_PATH / category / "manifest.csv"
        if not manifest_path.exists():
            print(f"⚠️  Manifest not found for {category}, skipping.")
            continue

        test_dataset = MVTecADDataset(manifest_path, split="test")

        print(f"\n{'='*60}")
        print(f"Category: {category} | Test set size: {len(test_dataset)}")
        print(f"{'='*60}")

        architectures = [
            ("resnet18", ResNet18Classifier),
            ("vit", ViTClassifier),
        ]

        for arch_name, ModelClass in architectures:
            ckpt_path = config.MODELS_PATH / arch_name / category / f"{arch_name}_best_val_acc.pth"

            if not ckpt_path.exists():
                print(f"  ⚠️  Checkpoint not found: {ckpt_path}, skipping.")
                continue

            model = ModelClass(pretrained=False, num_classes=2)  # pretrained=False, weights come from checkpoint
            model, _, epoch = load_checkpoint(model, None, ckpt_path, device)
            model = model.to(device)

            results = evaluate_on_test(model, test_dataset, device)
            summary = summarize_results(results, category, arch_name)
            all_summaries.append(summary)

            save_filtered_manifest(results, category, arch_name, output_dir)

    # Save overall summary CSV
    summary_path = output_dir / "evaluation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"\n{'='*60}")
    print(f"✨ Evaluation complete. Summary saved to: {summary_path}")
    print(f"{'='*60}")

    flagged = [s for s in all_summaries if s["flagged_low_sample"]]
    if flagged:
        print(f"\n⚠️  {len(flagged)} combination(s) flagged for low correctly-classified defective count:")
        for s in flagged:
            print(f"   - {s['category']} / {s['architecture']}: {s['n_correct_defective']} images")
        print("   Consider excluding these from XAI analysis, or noting as an architecture-dependent limitation.")
    else:
        print("\n✅ All category/architecture combinations have sufficient correctly-classified defective images.")


if __name__ == "__main__":
    main()