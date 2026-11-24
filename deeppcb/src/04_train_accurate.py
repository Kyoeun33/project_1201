"""Train full-image paired-input Faster R-CNN without candidate gating."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import make_accurate_model
from training import (PairDataset, collate_pairs, load_config, save_checkpoint,
                      select_device, set_seed, add_training_history, assess_overfitting)


def _batch_loss(model, images, targets, device):
    inputs = [image.to(device) for image in images]
    annotations = [{key: value.to(device) for key, value in target.items()} for target in targets]
    losses = model(inputs, annotations)
    return sum(losses.values())


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "accurate.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0, help="연결 점검용; 0이면 전체")
    parser.add_argument("--limit-val", type=int, default=0, help="연결 점검용; 0이면 전체")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg["device"])
    train = PairDataset(cfg, "train", augment=cfg["training"]["augmentation"], limit=args.limit_train)
    val = PairDataset(cfg, "val", limit=args.limit_val)
    train_loader = DataLoader(train, batch_size=cfg["training"]["batch_images"], shuffle=True,
                              num_workers=0, collate_fn=collate_pairs)
    val_loader = DataLoader(val, batch_size=cfg["training"]["batch_images"], shuffle=False,
                            num_workers=0, collate_fn=collate_pairs)
    model = make_accurate_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    checkpoint = Path(cfg["runs_dir"]) / "accurate.pt"
    best = float("inf")
    patience = int(cfg["training"].get("early_stopping_patience", 7))
    stale_epochs = 0
    train_history, val_history = [], []
    stopped_early = False
    for epoch in range(1, (args.epochs or int(cfg["training"]["epochs"])) + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, images, targets, device)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
        # Detection modules produce losses in train mode. Keep BN running stats
        # frozen during validation, including the shared backbone's BN layers.
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                val_loss += float(_batch_loss(model, images, targets, device).detach())
        train_mean = train_loss / len(train_loader)
        val_mean = val_loss / len(val_loader)
        train_history.append(train_mean)
        val_history.append(val_mean)
        print(f"epoch {epoch}: train={train_mean:.4f}, val={val_mean:.4f}, device={device}")
        if val_mean < best:
            best = val_mean
            stale_epochs = 0
            save_checkpoint(checkpoint, model, cfg, "accurate", epoch, best)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                print(f"Early stopping at epoch {epoch} (no validation improvement for {patience} epochs)")
                break
    add_training_history(checkpoint, train_history, val_history, stopped_early)
    print(f"Overfitting detected: {assess_overfitting(train_history, val_history)['detected']}")
    print(f"정밀형 체크포인트: {checkpoint}")


if __name__ == "__main__":
    main()
