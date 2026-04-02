"""
Dataset generator for modular addition: (a + b) mod p.

Generates all p^2 equations and provides train/test splits
controlled by a `fraction` parameter — the key hyperparameter
driving the grokking phase transition.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


PRIME = 97  # The modular base used in the original grokking paper


class ModularAdditionDataset(Dataset):
    """
    Each sample is (a, b) -> (a + b) % p.
    Inputs are provided as token indices; the label is the residue class.
    """

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: np.ndarray of shape (N, 3) where columns are [a, b, target].
        """
        self.a = torch.tensor(data[:, 0], dtype=torch.long)
        self.b = torch.tensor(data[:, 1], dtype=torch.long)
        self.targets = torch.tensor(data[:, 2], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input is a sequence of two tokens: [a, b]
        x = torch.stack([self.a[idx], self.b[idx]])
        y = self.targets[idx]
        return x, y


def generate_all_pairs(p: int = PRIME) -> np.ndarray:
    """
    Generate all p^2 pairs (a, b) and their targets (a + b) % p.

    Returns:
        np.ndarray of shape (p*p, 3) with columns [a, b, target].
    """
    pairs = []
    for a in range(p):
        for b in range(p):
            pairs.append([a, b, (a + b) % p])
    return np.array(pairs)


def get_dataloaders(
    fraction: float = 0.5,
    p: int = PRIME,
    batch_size: int = 512,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split the full dataset into train/test and return DataLoaders.

    Args:
        fraction: Proportion of data used for training (0.0 to 1.0).
                  This is the key hyperparameter controlling grokking onset.
        p:         The prime modulus.
        batch_size: Batch size for both loaders.
        seed:      Random seed for reproducible splits.

    Returns:
        (train_loader, test_loader)
    """
    all_data = generate_all_pairs(p)
    n_total = len(all_data)
    n_train = int(n_total * fraction)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)

    train_data = all_data[indices[:n_train]]
    test_data = all_data[indices[n_train:]]

    train_dataset = ModularAdditionDataset(train_data)
    test_dataset = ModularAdditionDataset(test_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    all_pairs = generate_all_pairs()
    print(f"Total equations: {len(all_pairs)}")  # Should be 9409
    print(f"Sample: {all_pairs[0]} => ({all_pairs[0][0]} + {all_pairs[0][1]}) mod {PRIME} = {all_pairs[0][2]}")

    train_loader, test_loader = get_dataloaders(fraction=0.5)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
