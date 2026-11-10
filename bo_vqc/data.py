"""Toy dataset: two interleaving moons, scaled to rotation angles."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def make_dataset(n_samples: int = 300, noise: float = 0.2, seed: int = 0, test_size: float = 1 / 3):
    """Return (X_train, y_train, X_test, y_test, scaler).

    Features are scaled to [-1, 1] so a multiplier of pi gives a full rotation
    range. Labels are in {-1, +1} to match the <Z> readout of the circuit.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = 2 * y - 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_tr)
    return scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te, scaler
