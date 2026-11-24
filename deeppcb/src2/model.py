"""One RGB input, two logits. Normalization and augmentation follow native tile extraction."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torchvision.models import (EfficientNet_B0_Weights, ResNet18_Weights,
                                efficientnet_b0, resnet18)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set training.device to auto or cpu.")
    return torch.device(name)


def network(name, pretrained):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return model


def input_batch(images, device, settings, augment=False):
    x = images.to(device, dtype=torch.float32, non_blocking=True).div_(255)
    if settings["preprocessing"] == "binary":
        gray = x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114
        x = (gray >= settings["binary_cutoff"] / 255.0).float().unsqueeze(1).expand(-1, 3, -1, -1)
    if augment:
        for axis in (2, 3):
            flip = torch.rand((len(x), 1, 1, 1), device=device) < 0.5
            x = torch.where(flip, x.flip((axis,)), x)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def batch_scores(model, images, device, settings):
    # Float32 inference is shared by validation, final evaluation and single-image prediction.
    logits = model(input_batch(images, device, settings))
    return logits.float().softmax(dim=1)[:, 1].cpu().numpy().astype(np.float64)


def score_loader(model, batches, device, settings):
    model.eval()
    return np.concatenate([batch_scores(model, images, device, settings) for images, _ in batches])


def restore(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = network(saved["training"]["model"], False)
    model.load_state_dict(saved["state_dict"])
    return model.to(device, memory_format=torch.channels_last).eval(), saved
