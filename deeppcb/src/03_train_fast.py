"""Train the fast XOR-candidate seven-class patch classifier."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import FastPatchCNN
from training import (FastCandidateDataset, collate_fast, load_config,
                      require_full_candidate_check, save_checkpoint, select_device, set_seed,
                      add_training_history, assess_overfitting)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "fast.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0, help="연결 점검용; 0이면 전체")
    parser.add_argument("--limit-val", type=int, default=0, help="연결 점검용; 0이면 전체")
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_full_candidate_check(cfg)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg["device"])
    train = FastCandidateDataset(cfg, "train", augment=cfg["training"]["augmentation"], limit=args.limit_train)
    val = FastCandidateDataset(cfg, "val", limit=args.limit_val)
    train_loader = DataLoader(train, batch_size=cfg["training"]["batch_images"], shuffle=True,
                              num_workers=0, collate_fn=collate_fast)
    val_loader = DataLoader(val, batch_size=cfg["training"]["batch_images"], shuffle=False,
                            num_workers=0, collate_fn=collate_fast)
    model = FastPatchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    weights = torch.tensor([0.5] + [1.0] * 6, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    checkpoint = Path(cfg["runs_dir"]) / "fast.pt"
    best = float("inf")
    patience = int(cfg["training"].get("early_stopping_patience", 7))
    stale_epochs = 0
    train_history, val_history = [], []
    stopped_early = False
    for epoch in range(1, (args.epochs or int(cfg["training"]["epochs"])) + 1):
        model.train()
        train_loss, train_count = 0.0, 0
        for patches, targets in train_loader:
            if not len(targets):
                continue
            patches, targets = patches.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(patches), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            train_count += len(targets)
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.inference_mode():
            for patches, targets in val_loader:
                if not len(targets):
                    continue
                patches, targets = patches.to(device), targets.to(device)
                loss = criterion(model(patches), targets)
                val_loss += loss.item() * len(targets)
                val_count += len(targets)
        if not train_count or not val_count:
            raise ValueError("학습 또는 검증 후보 패치가 0개")
        val_mean = val_loss / val_count
        train_mean = train_loss / train_count
        train_history.append(train_mean)
        val_history.append(val_mean)
        print(f"epoch {epoch}: train={train_mean:.4f}, val={val_mean:.4f}, device={device}")
        if val_mean < best:
            best = val_mean
            stale_epochs = 0
            save_checkpoint(checkpoint, model, cfg, "fast", epoch, best)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                print(f"Early stopping at epoch {epoch} (no validation improvement for {patience} epochs)")
                break
    add_training_history(checkpoint, train_history, val_history, stopped_early)
    print(f"Overfitting detected: {assess_overfitting(train_history, val_history)['detected']}")
    print(f"속도형 체크포인트: {checkpoint}")


if __name__ == "__main__":
    main()
