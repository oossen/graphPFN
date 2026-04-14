from pathlib import Path

import numpy as np


def save_array(path, array: np.ndarray) -> None:
    """Save a numpy array to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_array(path) -> np.ndarray:
    """Load a numpy array from a .npy file."""
    path = Path(path)
    assert path.exists(), f"File '{path}' not found."
    return np.load(path, allow_pickle=True)
