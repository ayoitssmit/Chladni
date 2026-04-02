"""
Dataset generator for modular arithmetic tasks.

Supports multiple operations on Z/pZ:
  - addition:      (a + b) mod p
  - subtraction:   (a - b) mod p
  - multiplication:(a * b) mod p
  - polynomial:    (a^2 + a*b + b^2) mod p
  - division:      (a * b^-1) mod p  (b=0 is excluded; no inverse exists)

The `fraction` parameter controls the train/test split and is the primary
hyperparameter that drives the grokking phase transition.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Literal

PRIME = 97  # The modular base used in the original grokking paper

# All supported operation names
Operation = Literal["addition", "subtraction", "multiplication", "polynomial", "division"]

# Human-readable labels for the frontend
OPERATION_LABELS: dict[str, str] = {
    "addition":       "(a + b) mod 97",
    "subtraction":    "(a - b) mod 97",
    "multiplication": "(a × b) mod 97",
    "polynomial":     "(a² + ab + b²) mod 97",
    "division":       "(a / b) mod 97",
}


def modular_inverse(b: int, p: int) -> int:
    """
    Compute the modular inverse of b under modulus p using Fermat's Little Theorem.
    Since p is prime: b^(-1) ≡ b^(p-2) (mod p).

    Args:
        b: The integer to invert. Must be non-zero.
        p: A prime modulus.

    Returns:
        b^(-1) mod p
    """
    assert b != 0, "Modular inverse is undefined for b=0"
    return pow(b, p - 2, p)


def compute_target(a: int, b: int, p: int, operation: Operation) -> int:
    """
    Compute the target label for a given (a, b) pair and operation.

    Args:
        a:         First operand (0 <= a < p)
        b:         Second operand (0 <= b < p)
        p:         Prime modulus
        operation: One of the supported operation strings

    Returns:
        The integer result in the range [0, p)
    """
    if operation == "addition":
        return (a + b) % p
    elif operation == "subtraction":
        return (a - b) % p
    elif operation == "multiplication":
        return (a * b) % p
    elif operation == "polynomial":
        return (a**2 + a * b + b**2) % p
    elif operation == "division":
        return (a * modular_inverse(b, p)) % p
    else:
        raise ValueError(f"Unknown operation: '{operation}'. "
                         f"Choose from: {list(OPERATION_LABELS.keys())}")


class ModularArithmeticDataset(Dataset):
    """
    A Dataset of (a, b) -> target pairs for modular arithmetic tasks.
    Each sample returns a two-token input tensor and a scalar label.
    """

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: np.ndarray of shape (N, 3) — columns are [a, b, target].
        """
        self.a = torch.tensor(data[:, 0], dtype=torch.long)
        self.b = torch.tensor(data[:, 1], dtype=torch.long)
        self.targets = torch.tensor(data[:, 2], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([self.a[idx], self.b[idx]])
        y = self.targets[idx]
        return x, y


def generate_all_pairs(p: int = PRIME, operation: Operation = "addition") -> np.ndarray:
    """
    Generate all valid (a, b) pairs and their targets for the given operation.

    Note: For 'division', pairs where b=0 are excluded because b^(-1) mod p
    is undefined. This gives p*(p-1) = 9,312 samples instead of p^2 = 9,409.

    Args:
        p:         The prime modulus.
        operation: The arithmetic operation to use.

    Returns:
        np.ndarray of shape (N, 3) with columns [a, b, target].
    """
    pairs = []
    for a in range(p):
        for b in range(p):
            # Skip b=0 for division: modular inverse is undefined
            if operation == "division" and b == 0:
                continue
            target = compute_target(a, b, p, operation)
            pairs.append([a, b, target])
    return np.array(pairs)


def get_dataloaders(
    fraction: float = 0.5,
    p: int = PRIME,
    operation: Operation = "addition",
    batch_size: int = 512,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split the full operation dataset into train/test and return DataLoaders.

    Args:
        fraction:   Proportion of data used for training (0.0 to 1.0).
                    This is the key hyperparameter controlling grokking onset.
        p:          The prime modulus.
        operation:  The arithmetic operation to train on.
        batch_size: Batch size for both loaders.
        seed:       Random seed for reproducible splits.

    Returns:
        (train_loader, test_loader)
    """
    all_data = generate_all_pairs(p, operation)
    n_total = len(all_data)
    n_train = int(n_total * fraction)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)

    train_data = all_data[indices[:n_train]]
    test_data = all_data[indices[n_train:]]

    train_dataset = ModularArithmeticDataset(train_data)
    test_dataset = ModularArithmeticDataset(test_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    print(f"Prime modulus: {PRIME}\n")

    for op in OPERATION_LABELS:
        all_pairs = generate_all_pairs(p=PRIME, operation=op)
        sample = all_pairs[7]  # Use index 7 to get a more interesting example
        print(f"[{op:>14}]  Total samples: {len(all_pairs):>5} | "
              f"Sample: {OPERATION_LABELS[op]} where a={sample[0]}, b={sample[1]} => {sample[2]}")

    # Verify division handles b=0 exclusion
    div_data = generate_all_pairs(p=PRIME, operation="division")
    assert len(div_data) == PRIME * (PRIME - 1), "Division should exclude b=0 pairs"

    # Verify manual calculation: 5 * inv(3) mod 97
    inv3 = modular_inverse(3, 97)
    result = (5 * inv3) % 97
    print(f"\nVerification: 5 / 3 mod 97 = {result}  (check: {result} * 3 mod 97 = {(result * 3) % 97})")

    # Spot-check dataloaders for each operation
    print("\nDataloader sizes (fraction=0.4):")
    for op in OPERATION_LABELS:
        train_loader, test_loader = get_dataloaders(fraction=0.4, operation=op)
        print(f"  {op:>14}: train={len(train_loader.dataset):>4}, "
              f"test={len(test_loader.dataset):>4}")
