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

from dataset import get_dataloaders, PRIME, Operation, OPERATION_LABELS
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

    Embeddings are L2-normalized to unit norm before PCA so that tokens with
    extreme magnitudes (e.g. token 0 in division, which is never a denominator)
    don't collapse or dominate the projection. The ring structure is in the
    *directions* of embeddings, not their magnitudes.

    Args:
        embeddings: (p, d_model) tensor
        n_components: Number of PCA dimensions (2 or 3)

    Returns:
        np.ndarray of shape (p, n_components)
    """
    emb_np = embeddings.cpu().numpy()
    # L2-normalize each token vector to unit norm
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)  # guard against zero vectors
    emb_np = emb_np / norms
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(emb_np)
    return reduced


def train_generator(
    fraction: float = 0.5,
    weight_decay: float = 1.0,
    lr: float = 1e-3,
    total_steps: int = 50000,
    log_every: int = 100,
    operation: Operation = "addition",
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 200,
    n_layers: int = 1,
    seed: int = 42,
    shared_state: Dict[str, bool] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields training metrics at regular intervals.

    Each yield produces a dict:
    {
        "step": int,
        "operation": str,
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
        operation:    The arithmetic task to train on (addition, subtraction,
                      multiplication, polynomial, division)
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
    print(f"[train] Operation: {OPERATION_LABELS.get(operation, operation)}")
    train_loader, test_loader = get_dataloaders(
        fraction=fraction, p=PRIME, operation=operation, batch_size=512, seed=seed
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
    test_acc_history = []
    locked_prediction = -1
    intervention_triggered = False
    
    if shared_state is None:
        shared_state = {"intervene": False}

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
            
            # Apply manual intervention if requested via shared_state
            if shared_state.get("intervene", False) and not intervention_triggered:
                intervention_triggered = True
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] *= 5.0
                print(f"[intervention] Weight decay sharply increased to {weight_decay * 5.0} at step {step}!")

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

                # ── FFT Early Warning System ──
                # Simple threshold detector on dominant Fourier frequency power.
                # The insight: Fourier features grow slowly for hundreds of steps
                # before test accuracy jumps. We monitor that growth and predict.
                emb_centered = emb.detach().cpu()
                emb_centered = emb_centered - emb_centered.mean(dim=0)
                fft_vals = torch.fft.fft(emb_centered, dim=0)
                power_spectrum = (fft_vals.abs() ** 2).sum(dim=1)  # [p]
                # Dominant non-DC frequency power, normalized
                total_power = float(power_spectrum.sum().item()) + 1e-12
                # Dominant non-DC frequency power (kept for UI signal bar)
                total_power = float(power_spectrum.sum().item()) + 1e-12
                dominant_power = float(power_spectrum[1:].max().item())
                fft_signal = dominant_power / total_power
                
                # ── Predictive Model: Sigmoid / Logit Extrapolation ──
                # Test accuracy in grokking follows a very sharp S-Curve (Sigmoid).
                # Linear extrapolation on an S-curve produces massive overestimations (e.g. 11k steps).
                # The standard statistical approach is to use the Logit transform:
                # z = ln(p / (1 - p)), which magically turns the S-curve into a straight line!
                test_acc_history.append((step, test_acc))

                # Baseline accuracy for random guessing mod 97 is ~1.03%
                # We only start fitting when test accuracy reliably breaks 1.5%
                valid_history = [(s, max(1e-4, min(1-1e-4, a))) for s, a in test_acc_history if a >= 0.015]

                if not grokked and len(valid_history) >= 2:
                    recent = valid_history[-5:]
                    if len(recent) >= 2:
                        X = np.array([pt[0] for pt in recent])
                        Y = np.array([pt[1] for pt in recent])
                        
                        # Logit transformation straightens the S-Curve
                        Z = np.log(Y / (1.0 - Y))
                        m, c = np.polyfit(X, Z, 1)
                        
                        # Once the steep grokking jump actually begins
                        if m > 0.001:
                            z_target = np.log(0.95 / 0.05)
                            pred = (z_target - c) / m
                            
                            if step < pred < total_steps * 2:
                                new_pred = int(pred)
                                if locked_prediction == -1:
                                    locked_prediction = new_pred
                                else:
                                    # Continuously refine the prediction seamlessly.
                                    # We only update if the new prediction is CLOSER,
                                    # guaranteeing the UI countdown strictly shrinks.
                                    if new_pred < locked_prediction:
                                        locked_prediction = new_pred
                                        
                                print(f"[predict] Step {step}: Logit slope={m:.6f}, "
                                      f"Refined grok ~{locked_prediction}")

                predicted_grok_step = locked_prediction if not grokked else -1

                elapsed = time.time() - start_time

                payload = {
                    "step": step,
                    "total_steps": total_steps,
                    "operation": operation,
                    "train_loss": round(float(loss.item()), 6),
                    "train_accuracy": round(train_acc, 4),
                    "test_accuracy": round(test_acc, 4),
                    "pca_embeddings": pca_coords.tolist(),
                    "grokked": grokked,
                    "elapsed_seconds": round(elapsed, 2),
                    "fft_signal": round(fft_signal * 100, 2),  # 0-100%
                    "predicted_grok_step": predicted_grok_step,
                    "intervention_triggered": intervention_triggered,
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
        "operation": operation,
        "train_loss": round(float(loss.item()), 6),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "pca_embeddings": pca_coords.tolist(),
        "grokked": grokked,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "fft_signal": round(fft_signal * 100, 2) if 'fft_signal' in locals() else 0.0,
        "predicted_grok_step": -1,
        "intervention_triggered": intervention_triggered,
        "finished": True,
    }


def train_single_run(
    fraction: float = 0.5,
    weight_decay: float = 1.0,
    total_steps: int = 50000,
    log_every: int = 500,
    operation: Operation = "addition",
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
        operation=operation,
        **kwargs,
    ):
        last_payload = payload
        if payload["grokked"] and grok_step == -1:
            grok_step = payload["step"]

    if last_payload:
        last_payload["grok_step"] = grok_step

    return last_payload


if __name__ == "__main__":
    import sys
    op = sys.argv[1] if len(sys.argv) > 1 else "addition"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    
    print("=" * 60)
    print(f"Training demo — Operation: {op} | Steps: {steps}")
    print("=" * 60)

    for snapshot in train_generator(
        fraction=0.4,
        weight_decay=1.0,
        total_steps=steps,
        log_every=500,
        operation=op,
    ):
        step = snapshot["step"]
        tr = snapshot["train_accuracy"]
        te = snapshot["test_accuracy"]
        loss = snapshot["train_loss"]
        g = " GROKKED" if snapshot["grokked"] else ""
        print(f"Step {step:>6} | Loss: {loss:.4f} | Train: {tr:.2%} | Test: {te:.2%} {g}")
