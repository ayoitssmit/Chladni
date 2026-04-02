"""
Training loop with real-time metric extraction and PCA embedding tracking.

This module orchestrates:
  1. Model training with AdamW optimizer
  2. Train/test accuracy tracking per step
  3. Token embedding extraction → PCA reduction to 2D/3D
  4. Attention weight capture via model hooks
  5. A generator-based API for streaming results step-by-step
"""

import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Any, Generator, Optional

from dataset import get_dataloaders, PRIME
from model import GrokTransformer


def compute_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    """Compute classification accuracy over a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def embeddings_to_pca(
    embeddings: torch.Tensor, n_components: int = 3
) -> np.ndarray:
    """
    Reduce token embeddings from (p, d_model) to (p, n_components) via PCA.

    Args:
        embeddings: (p, d_model) tensor
        n_components: Number of PCA dimensions (2 or 3)

    Returns:
        np.ndarray of shape (p, n_components)
    """
    emb_np = embeddings.cpu().numpy()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(emb_np)
    return reduced


def train_generator(
    fraction: float = 0.5,
    weight_decay: float = 1.0,
    lr: float = 1e-3,
    total_steps: int = 50000,
    log_every: int = 100,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 200,
    n_layers: int = 1,
    seed: int = 42,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields training metrics at regular intervals.

    Each yield produces a dict:
    {
        "step": int,
        "train_loss": float,
        "train_accuracy": float,
        "test_accuracy": float,
        "pca_embeddings": [[x, y, z], ...] (97 points),
        "grokked": bool,
        "elapsed_seconds": float,
    }

    Args:
        fraction:     Proportion of data used for training
        weight_decay: L2 regularization strength (key grokking driver)
        lr:           Learning rate
        total_steps:  Maximum training steps
        log_every:    Steps between metric snapshots
        d_model:      Transformer hidden dimension
        n_heads:      Number of attention heads
        d_ff:         Feed-forward hidden dimension
        n_layers:     Number of transformer layers
        seed:         Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, test_loader = get_dataloaders(
        fraction=fraction, p=PRIME, batch_size=512, seed=seed
    )

    # Model
    model = GrokTransformer(
        p=PRIME,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
    ).to(device)

    print(f"[train] Parameters: {model.count_parameters():,}")
    print(f"[train] Device: {device}")
    print(f"[train] Train samples: {len(train_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    # Optimizer — AdamW with configurable weight_decay is the grokking secret
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )

    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    step = 0
    start_time = time.time()
    grokked = False
    best_test_acc = 0.0

    while step < total_steps:
        for x, y in train_loader:
            if step >= total_steps:
                break

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            step += 1

            # Emit metrics at regular intervals
            if step % log_every == 0 or step == 1:
                train_acc = compute_accuracy(model, train_loader, device)
                test_acc = compute_accuracy(model, test_loader, device)

                # PCA of token embeddings
                emb = model.get_token_embeddings()
                pca_coords = embeddings_to_pca(emb, n_components=3)

                # Detect grokking: test accuracy crosses 95% after
                # training accuracy was already > 99%
                if test_acc > 0.95 and train_acc > 0.99 and not grokked:
                    grokked = True
                    print(f"[train] *** GROKKING DETECTED at step {step}! ***")

                elapsed = time.time() - start_time

                payload = {
                    "step": step,
                    "total_steps": total_steps,
                    "train_loss": round(float(loss.item()), 6),
                    "train_accuracy": round(train_acc, 4),
                    "test_accuracy": round(test_acc, 4),
                    "pca_embeddings": pca_coords.tolist(),
                    "grokked": grokked,
                    "elapsed_seconds": round(elapsed, 2),
                }

                yield payload

    # Final snapshot
    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    emb = model.get_token_embeddings()
    pca_coords = embeddings_to_pca(emb, n_components=3)

    yield {
        "step": step,
        "total_steps": total_steps,
        "train_loss": round(float(loss.item()), 6),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "pca_embeddings": pca_coords.tolist(),
        "grokked": grokked,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "finished": True,
    }


def train_single_run(
    fraction: float = 0.5,
    weight_decay: float = 1.0,
    total_steps: int = 50000,
    log_every: int = 500,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a full training session and return the final result.
    Useful for batch runs (phase diagram generation).

    Returns:
        Dict with final metrics + grok_step (step at which grokking happened, or -1)
    """
    grok_step = -1
    last_payload = None

    for payload in train_generator(
        fraction=fraction,
        weight_decay=weight_decay,
        total_steps=total_steps,
        log_every=log_every,
        **kwargs,
    ):
        last_payload = payload
        if payload["grokked"] and grok_step == -1:
            grok_step = payload["step"]

    if last_payload:
        last_payload["grok_step"] = grok_step

    return last_payload


if __name__ == "__main__":
    print("=" * 60)
    print("Running a quick training demo (5000 steps)...")
    print("=" * 60)

    for snapshot in train_generator(
        fraction=0.4,
        weight_decay=1.0,
        total_steps=5000,
        log_every=500,
    ):
        step = snapshot["step"]
        tr = snapshot["train_accuracy"]
        te = snapshot["test_accuracy"]
        loss = snapshot["train_loss"]
        g = " GROKKED" if snapshot["grokked"] else ""
        print(f"Step {step:>6} | Loss: {loss:.4f} | Train: {tr:.2%} | Test: {te:.2%} {g}")
