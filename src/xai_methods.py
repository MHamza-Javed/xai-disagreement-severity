"""
XAI Methods Implementation
- Grad-CAM (ResNet-18: standard conv-layer Grad-CAM; ViT: custom adaptation
  using the second-to-last transformer block's token outputs, reshaped to a spatial grid)
- Integrated Gradients
- Occlusion Sensitivity
- GradientSHAP

All model-agnostic methods (IG, Occlusion, GradientSHAP) via PyTorch Captum.
Grad-CAM is architecture-specific due to ViT having no convolutional layers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Occlusion, GradientShap, LayerGradCam
from captum.attr import LayerAttribution


def _normalize(heatmap):
    """Normalize a heatmap to [0, 1]."""
    heatmap = heatmap - heatmap.min()
    max_val = heatmap.max()
    if max_val > 1e-8:
        heatmap = heatmap / max_val
    return heatmap


def _reduce_channels(attr):
    """
    Reduce a [1, C, H, W] attribution tensor to a [H, W] heatmap
    by averaging absolute attribution across channels.
    """
    attr = attr.squeeze(0)  # [C, H, W]
    heatmap = attr.abs().mean(dim=0)  # [H, W]
    return heatmap.detach().cpu().numpy()


class XAIAnalyzer:
    """
    Wrapper for multi-method XAI analysis on a single model.

    Args:
        model (nn.Module): ResNet18Classifier or ViTClassifier (already loaded + eval mode)
        architecture (str): 'resnet18' or 'vit' — determines Grad-CAM strategy
        device (str): 'cpu' or 'cuda'
        img_size (int): Input image size (default 224)
        patch_size (int): ViT patch size, needed to reshape tokens back to a grid (default 16)
    """

    def __init__(self, model, architecture, device='cpu', img_size=224, patch_size=16):
        self.model = model.to(device)
        self.model.eval()
        self.architecture = architecture
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g. 224/16 = 14

        # Model-agnostic methods work the same regardless of architecture
        self.integrated_gradients = IntegratedGradients(self.model)
        self.occlusion = Occlusion(self.model)
        self.gradient_shap = GradientShap(self.model)

        if architecture == 'resnet18':
            # Standard Grad-CAM on the last conv block
            target_layer = self.model.model.layer4[-1]
            self.gradcam = LayerGradCam(self.model, target_layer)
        elif architecture == 'vit':
            # No conv layer to hook — Grad-CAM computed manually via forward/backward hooks
            self._vit_activations = None
            self._vit_gradients = None
            self._register_vit_hooks()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    # ---------------- ViT Grad-CAM adaptation ----------------

    def _register_vit_hooks(self):
        """
        Register forward/backward hooks on the second-to-last transformer block.
        (Hooking the LAST block doesn't work: only its CLS-token output feeds the
        head, so its patch-token gradients are structurally zero.)
        """
        target_block = self.model.model.blocks[-2]

        def forward_hook(module, input, output):
            self._vit_activations = output

        def backward_hook(module, grad_input, grad_output):
            self._vit_gradients = grad_output[0]

        target_block.register_forward_hook(forward_hook)
        target_block.register_full_backward_hook(backward_hook)

    def _compute_vit_gradcam(self, image, target_class):
        """
        Custom Grad-CAM for ViT: weight the block's patch-token activations
        by their gradients w.r.t. the target class, then reshape the resulting
        per-token importance back into a 2D spatial grid (excluding the CLS token).
        """
        # Reset captured tensors so we can tell if the hooks actually fired this call
        self._vit_activations = None
        self._vit_gradients = None

        image = image.clone().detach().requires_grad_(True)
        output = self.model(image)

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # --- DIAGNOSTIC: did the hooks fire at all? ---
        if self._vit_activations is None:
            raise RuntimeError("Forward hook never fired — _vit_activations is None. "
                                "The target block was not reached during forward().")
        if self._vit_gradients is None:
            raise RuntimeError("Backward hook never fired — _vit_gradients is None. "
                                "Gradient never flowed back to the target block.")

        activations = self._vit_activations[0]  # [num_tokens, dim]
        gradients = self._vit_gradients[0]       # [num_tokens, dim]

        # Pool gradients across the TOKEN axis (the ViT analog of "spatial location" in
        # CNN Grad-CAM) to get one weight per feature CHANNEL — NOT the other way around.
        # Averaging across the feature dim per-token hits an exact-zero identity: the
        # next block starts with LayerNorm, whose backward pass guarantees gradients
        # sum to zero across the feature dimension.
        weights = gradients.mean(dim=0)  # [dim] — one weight per channel

        # Weighted combination across channels, for each token separately
        token_importance = (activations * weights.unsqueeze(0)).sum(dim=-1)  # [num_tokens]
        token_importance = token_importance.abs()

        token_importance = token_importance.abs()

        # Drop CLS token (index 0), reshape remaining patch tokens to grid
        patch_importance = token_importance[1:]  # [grid*grid]
        if patch_importance.numel() != self.grid_size * self.grid_size:
            raise ValueError(
                f"Unexpected token count {patch_importance.numel()} for grid "
                f"{self.grid_size}x{self.grid_size} — check patch_size/img_size."
            )

        heatmap = patch_importance.reshape(self.grid_size, self.grid_size)
        heatmap = heatmap.detach().cpu().numpy()

        # Upsample from grid_size x grid_size to full image resolution
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_t = F.interpolate(heatmap_t, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)
        heatmap = heatmap_t.squeeze().numpy()

        return _normalize(heatmap)

    # ---------------- Public method: Grad-CAM (dispatches by architecture) ----------------

    def compute_gradcam(self, image, target_class):
        """
        Compute Grad-CAM heatmap.

        Args:
            image (Tensor): Input image [1, C, H, W], on self.device
            target_class (int): Target class index for attribution

        Returns:
            heatmap (ndarray): [H, W] normalized heatmap in [0, 1]
        """
        image = image.to(self.device)

        if self.architecture == 'resnet18':
            attr = self.gradcam.attribute(image, target=target_class)
            attr = LayerAttribution.interpolate(attr, (self.img_size, self.img_size))
            heatmap = attr.squeeze().detach().cpu().numpy()
            return _normalize(heatmap)
        else:  # vit
            return self._compute_vit_gradcam(image, target_class)

    # ---------------- Model-agnostic methods ----------------

    def compute_integrated_gradients(self, image, target_class, n_steps=50):
        """Compute Integrated Gradients heatmap (baseline = black image)."""
        image = image.to(self.device)
        baseline = torch.zeros_like(image)
        attr = self.integrated_gradients.attribute(
            image, baselines=baseline, target=target_class, n_steps=n_steps
        )
        heatmap = _reduce_channels(attr)
        return _normalize(heatmap)

    def compute_occlusion_sensitivity(self, image, target_class,
                                       sliding_window_shapes=(3, 28, 28), strides=(3, 14, 14)):
        """Compute Occlusion Sensitivity heatmap."""
        image = image.to(self.device)
        baseline = torch.zeros_like(image)
        attr = self.occlusion.attribute(
            image, sliding_window_shapes=sliding_window_shapes, strides=strides,
            baselines=baseline, target=target_class
        )
        heatmap = _reduce_channels(attr)
        return _normalize(heatmap)

    def compute_gradient_shap(self, image, target_class, n_samples=50, stdevs=0.09):
        """Compute GradientSHAP heatmap (baseline distribution = random noise around zero)."""
        image = image.to(self.device)
        baseline_dist = torch.cat([
            torch.zeros_like(image),
            torch.randn_like(image) * stdevs,
        ])
        attr = self.gradient_shap.attribute(
            image, baselines=baseline_dist, target=target_class,
            n_samples=n_samples, stdevs=stdevs
        )
        heatmap = _reduce_channels(attr)
        return _normalize(heatmap)

    # ---------------- Convenience: all 4 methods at once ----------------

    def get_all_heatmaps(self, image, target_class):
        """
        Generate all four heatmaps for a single image.

        Returns:
            dict: {'gradcam': hm, 'ig': hm, 'occlusion': hm, 'gradientsshap': hm}
            each hm is a [H, W] ndarray normalized to [0, 1]
        """
        return {
            'gradcam': self.compute_gradcam(image, target_class),
            'ig': self.compute_integrated_gradients(image, target_class),
            'occlusion': self.compute_occlusion_sensitivity(image, target_class),
            'gradientsshap': self.compute_gradient_shap(image, target_class),
        }